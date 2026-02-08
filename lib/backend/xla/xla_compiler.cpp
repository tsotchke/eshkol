/*
 * XLA Compiler Implementation for Eshkol
 *
 * In LLVM-direct mode, the compiler operates as a thin pass-through: the
 * LLVM module IS the executable, so "compilation" means verifying that the
 * module is well-formed and ready for JIT or AOT emission. No separate XLA
 * compilation step is needed on the CPU path.
 *
 * GPU compilation (CUDA, Metal, Vulkan) is handled at the runtime dispatch
 * level rather than here. When a GPU target becomes active, this compiler
 * would route through the StableHLO emitter and XLA compilation pipeline.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/backend/xla/xla_compiler.h"
#include "eshkol/backend/xla/xla_codegen.h"

namespace eshkol {
namespace xla {

// ===== XLACompiler Implementation =====

class XLACompiler::Impl {
public:
    bool initialized_ = false;
};

XLACompiler::XLACompiler()
    : impl_(std::make_unique<Impl>()) {}

XLACompiler::~XLACompiler() = default;

// ===== Compilation =====

/// Compile a module for the given target and options.
/// In LLVM-direct mode, the LLVM module is the executable -- no separate
/// XLA compilation pass is needed. The module pointer is returned as the
/// executable handle, and success is reported immediately.
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

    impl_->initialized_ = true;

    // LLVM-direct mode: the LLVM module IS the executable.
    // No separate compilation step required for CPU target.
    return CompileResult{
        .success = true,
        .executable = module,
        .error_message = "",
        .compile_time_ms = 0,
        .executable_size_bytes = 0
    };
}

/// Compile a module for a specific target with default options.
CompileResult XLACompiler::compileForTarget(void* module, Target target) {
    CompileOptions options;
    options.target = target;
    return compile(module, options);
}

// ===== Target Queries =====

/// Check if a compilation target is available on this system.
/// CPU is always available. GPU targets are available when the corresponding
/// runtime backend is detected (handled at the runtime dispatch level).
bool XLACompiler::isTargetAvailable(Target target) {
    switch (target) {
        case Target::CPU:
            return true;
        case Target::CUDA:
        case Target::Metal:
        case Target::Vulkan:
            // GPU availability is determined at runtime dispatch level,
            // not at the compiler level. Return false here; the runtime
            // layer (gpu_memory, xla_codegen) handles GPU detection.
            return false;
    }
    return false;
}

/// Return the default compilation target for this system.
/// Always CPU -- GPU dispatch is handled at the operation level.
Target XLACompiler::getDefaultTarget() {
    return Target::CPU;
}

/// Return all targets available for compilation on this system.
/// Returns CPU. GPU targets are added dynamically by the runtime layer.
std::vector<Target> XLACompiler::getAvailableTargets() {
    return {Target::CPU};
}

// ===== Executable Management =====

/// Free a compiled executable.
/// In LLVM-direct mode, the arena manages all memory, so this is a no-op.
/// The LLVM module lifetime is owned by the codegen pipeline.
void XLACompiler::freeExecutable(void* executable) {
    (void)executable;
    // Arena manages memory; no explicit free needed.
}

/// Serialize a compiled executable to bytes for caching.
/// LLVM-direct mode does not produce standalone serializable executables;
/// the LLVM module itself is used directly. Returns empty to indicate
/// serialization is not applicable in this mode.
std::vector<uint8_t> XLACompiler::serializeExecutable(void* executable) {
    (void)executable;
    // LLVM-direct mode: module is used in-process, not serialized.
    return {};
}

/// Deserialize an executable from bytes.
/// Not applicable in LLVM-direct mode. Returns nullptr.
void* XLACompiler::deserializeExecutable(const std::vector<uint8_t>& data, Target target) {
    (void)data;
    (void)target;
    // LLVM-direct mode does not support executable deserialization.
    return nullptr;
}

} // namespace xla
} // namespace eshkol
