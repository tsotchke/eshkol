/*
 * XLA Compiler Implementation for Eshkol
 *
 * Dual-mode compilation:
 *
 * 1. StableHLO mode (ESHKOL_XLA_FULL_MLIR): Lowers StableHLO IR through
 *    the MLIR pass pipeline: StableHLO → Linalg → SCF → LLVM Dialect → LLVM IR.
 *    This enables MLIR optimizations (fusion, tiling, vectorization) before
 *    generating native code.
 *
 * 2. LLVM-direct mode (default): The LLVM module IS the executable. No
 *    separate compilation step is needed — operations are emitted directly
 *    as LLVM IR calling C runtime functions with BLAS/SIMD/GPU dispatch.
 *
 * GPU compilation (CUDA, Metal, Vulkan) is handled at the runtime dispatch
 * level in both modes. The StableHLO path additionally enables XLA-native
 * GPU compilation when the full XLA infrastructure is present.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/backend/xla/xla_compiler.h"
#include "eshkol/backend/xla/xla_codegen.h"
#include "eshkol/backend/gpu/gpu_memory.h"
#include <chrono>

// MLIR includes for the StableHLO → LLVM lowering pipeline
#if defined(ESHKOL_MLIR_AVAILABLE) && defined(ESHKOL_STABLEHLO_AVAILABLE)
#define ESHKOL_XLA_FULL_MLIR 1
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

// StableHLO → Linalg lowering
#include "stablehlo/conversions/linalg/transforms/Passes.h"

// Linalg → loops
#include "mlir/Dialect/Linalg/Passes.h"

// SCF → control flow
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"

// Standard to LLVM dialect
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"

// MLIR → LLVM IR translation
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#endif

namespace eshkol {
namespace xla {

// ===== XLACompiler Implementation =====

class XLACompiler::Impl {
public:
    bool initialized_ = false;

#ifdef ESHKOL_XLA_FULL_MLIR
    /// Run the MLIR pass pipeline: StableHLO → Linalg → SCF → LLVM Dialect
    bool runPassPipeline(mlir::ModuleOp module) {
        auto& ctx = *module.getContext();
        mlir::PassManager pm(&ctx);

        // Phase 1: StableHLO → Linalg (tensor → buffer)
        pm.addPass(mlir::stablehlo::createStablehloLegalizeToLinalgPass());

        // Phase 2: Linalg → loops (SCF structured control flow)
        pm.addNestedPass<mlir::func::FuncOp>(
            mlir::createConvertLinalgToLoopsPass());

        // Phase 3: SCF → cf (unstructured control flow)
        pm.addPass(mlir::createSCFToControlFlowPass());

        // Phase 4: Standard dialects → LLVM dialect
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        pm.addPass(mlir::createArithToLLVMConversionPass());
        pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());

        // Phase 5: Clean up unrealized conversion casts
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());

        return mlir::succeeded(pm.run(module));
    }
#endif
};

XLACompiler::XLACompiler()
    : impl_(std::make_unique<Impl>()) {}

XLACompiler::~XLACompiler() = default;

// ===== Compilation =====

/// Compile a module for the given target and options.
/// In StableHLO mode: lowers through MLIR pipeline to LLVM IR.
/// In LLVM-direct mode: pass-through (LLVM module IS the executable).
CompileResult XLACompiler::compile(void* module, const CompileOptions& options) {
    (void)options;

    if (!module) {
        return CompileResult{
            .success = false,
            .executable = nullptr,
            .error_message = "null module passed to XLA compiler",
            .compile_time_ms = 0,
            .executable_size_bytes = 0
        };
    }

#ifdef ESHKOL_XLA_FULL_MLIR
    auto start = std::chrono::high_resolution_clock::now();

    // The module is an mlir::ModuleOp* cast to void*
    auto* mlirModule = static_cast<mlir::ModuleOp*>(module);

    // Run the StableHLO → LLVM lowering pipeline
    if (!impl_->runPassPipeline(*mlirModule)) {
        return CompileResult{
            .success = false,
            .executable = nullptr,
            .error_message = "MLIR StableHLO lowering pipeline failed",
            .compile_time_ms = 0,
            .executable_size_bytes = 0
        };
    }

    // Register LLVM dialect translation for final export
    auto& ctx = *mlirModule->getContext();
    mlir::registerLLVMDialectTranslation(ctx);

    // Translate MLIR LLVM dialect → LLVM IR module
    llvm::LLVMContext llvmCtx;
    auto llvmModule = mlir::translateModuleToLLVMIR(*mlirModule, llvmCtx);
    if (!llvmModule) {
        return CompileResult{
            .success = false,
            .executable = nullptr,
            .error_message = "MLIR to LLVM IR translation failed",
            .compile_time_ms = 0,
            .executable_size_bytes = 0
        };
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    impl_->initialized_ = true;

    return CompileResult{
        .success = true,
        .executable = static_cast<void*>(llvmModule.release()),
        .error_message = "",
        .compile_time_ms = ms,
        .executable_size_bytes = 0
    };
#else
    // LLVM-direct mode: the LLVM module IS the executable.
    // No separate compilation step required for CPU target.
    impl_->initialized_ = true;

    return CompileResult{
        .success = true,
        .executable = module,
        .error_message = "",
        .compile_time_ms = 0,
        .executable_size_bytes = 0
    };
#endif
}

/// Compile a module for a specific target with default options.
CompileResult XLACompiler::compileForTarget(void* module, Target target) {
    CompileOptions options;
    options.target = target;
    return compile(module, options);
}

// ===== Target Queries =====

/// Check if a compilation target is available on this system.
/// CPU is always available. GPU targets check actual hardware via gpu_memory.
bool XLACompiler::isTargetAvailable(Target target) {
    if (target == Target::CPU) return true;
    // Ensure GPU subsystem is initialized before checking availability
    static bool gpu_checked = false;
    if (!gpu_checked) {
        gpu_checked = true;
        eshkol_gpu_init();
    }
    switch (target) {
        case Target::CUDA:   return eshkol_gpu_backend_available(ESHKOL_GPU_CUDA) != 0;
        case Target::Metal:  return eshkol_gpu_backend_available(ESHKOL_GPU_METAL) != 0;
        case Target::Vulkan: return eshkol_gpu_backend_available(ESHKOL_GPU_VULKAN) != 0;
        default: return false;
    }
}

/// Return the default compilation target for this system.
/// Returns GPU if available, otherwise CPU.
Target XLACompiler::getDefaultTarget() {
    // Ensure GPU subsystem is initialized before choosing default target
    static bool gpu_checked = false;
    if (!gpu_checked) {
        gpu_checked = true;
        eshkol_gpu_init();
    }
    EshkolGPUBackend backend = eshkol_gpu_get_backend();
    switch (backend) {
        case ESHKOL_GPU_METAL:  return Target::Metal;
        case ESHKOL_GPU_CUDA:   return Target::CUDA;
        case ESHKOL_GPU_VULKAN: return Target::Vulkan;
        default:                return Target::CPU;
    }
}

/// Return all targets available for compilation on this system.
/// Always includes CPU, plus any detected GPU backends.
std::vector<Target> XLACompiler::getAvailableTargets() {
    // Ensure GPU subsystem is initialized before checking
    static bool gpu_checked = false;
    if (!gpu_checked) {
        gpu_checked = true;
        eshkol_gpu_init();
    }
    std::vector<Target> targets = {Target::CPU};
    if (eshkol_gpu_backend_available(ESHKOL_GPU_METAL))
        targets.push_back(Target::Metal);
    if (eshkol_gpu_backend_available(ESHKOL_GPU_CUDA))
        targets.push_back(Target::CUDA);
    if (eshkol_gpu_backend_available(ESHKOL_GPU_VULKAN))
        targets.push_back(Target::Vulkan);
    return targets;
}

// ===== Executable Management =====

/// Free a compiled executable.
/// In LLVM-direct mode, the arena manages all memory, so this is a no-op.
/// In StableHLO mode, the LLVM module was heap-allocated by translateModuleToLLVMIR.
void XLACompiler::freeExecutable(void* executable) {
#ifdef ESHKOL_XLA_FULL_MLIR
    // StableHLO mode: executable is an llvm::Module* from translateModuleToLLVMIR
    if (executable) {
        delete static_cast<llvm::Module*>(executable);
    }
#else
    (void)executable;
    // LLVM-direct: arena manages memory; no explicit free needed.
#endif
}

/// Serialize a compiled executable to bytes for caching.
/// TODO: XLA JIT cache — wire into compilation pipeline when StableHLO is active.
/// StableHLO mode: serializes the MLIR module as textual IR.
/// LLVM-direct mode: not applicable (empty return).
std::vector<uint8_t> XLACompiler::serializeExecutable(void* executable) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!executable) return {};
    // Serialize the LLVM module to bitcode
    auto* llvmModule = static_cast<llvm::Module*>(executable);
    std::string str;
    llvm::raw_string_ostream os(str);
    llvmModule->print(os, nullptr);
    return std::vector<uint8_t>(str.begin(), str.end());
#else
    (void)executable;
    return {};
#endif
}

/// Deserialize an executable from bytes.
/// TODO: XLA JIT cache — implement with llvm::parseIR() for StableHLO mode.
/// Not yet implemented for StableHLO mode — would require parsing LLVM IR.
/// Returns nullptr in LLVM-direct mode.
void* XLACompiler::deserializeExecutable(const std::vector<uint8_t>& data, Target target) {
    (void)data;
    (void)target;
    // Not yet implemented. Would need llvm::parseIR() for StableHLO mode.
    return nullptr;
}

} // namespace xla
} // namespace eshkol
