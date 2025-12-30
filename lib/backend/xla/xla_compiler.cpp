/*
 * XLA Compiler Implementation for Eshkol
 *
 * STUB IMPLEMENTATION - Phase 1
 * This is a skeleton that compiles but doesn't use XLA yet.
 * Actual XLA compilation will be added in Phase 2.
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
    // TODO: Add XLA compilation infrastructure in Phase 2
};

XLACompiler::XLACompiler()
    : impl_(std::make_unique<Impl>()) {}

XLACompiler::~XLACompiler() = default;

// ===== Compilation (Stubs) =====

CompileResult XLACompiler::compile(void* module, const CompileOptions& options) {
    (void)module;
    (void)options;
    return CompileResult{
        .success = false,
        .executable = nullptr,
        .error_message = "XLA compilation not yet implemented (stub)",
        .compile_time_ms = 0,
        .executable_size_bytes = 0
    };
}

CompileResult XLACompiler::compileForTarget(void* module, Target target) {
    CompileOptions options;
    options.target = target;
    return compile(module, options);
}

// ===== Target Queries =====

bool XLACompiler::isTargetAvailable(Target target) {
    // STUB: Only CPU is "available" (but not functional yet)
    return target == Target::CPU;
}

Target XLACompiler::getDefaultTarget() {
    return Target::CPU;
}

std::vector<Target> XLACompiler::getAvailableTargets() {
    // STUB: Return CPU as the only available target
    return {Target::CPU};
}

// ===== Executable Management (Stubs) =====

void XLACompiler::freeExecutable(void* executable) {
    (void)executable;
    // STUB: Nothing to free yet
}

std::vector<uint8_t> XLACompiler::serializeExecutable(void* executable) {
    (void)executable;
    // STUB: Return empty
    return {};
}

void* XLACompiler::deserializeExecutable(const std::vector<uint8_t>& data, Target target) {
    (void)data;
    (void)target;
    // STUB: Return nullptr
    return nullptr;
}

} // namespace xla
} // namespace eshkol
