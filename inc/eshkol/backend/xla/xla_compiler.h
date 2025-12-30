/*
 * XLA Compiler Interface for Eshkol
 *
 * Compiles StableHLO IR to executable code via XLA backends.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_XLA_COMPILER_H
#define ESHKOL_XLA_COMPILER_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace eshkol {
namespace xla {

// Forward declaration
enum class Target;

/**
 * Optimization level for XLA compilation
 */
enum class OptLevel {
    O0,  // No optimization (fastest compile)
    O1,  // Basic optimization
    O2,  // Standard optimization (default)
    O3   // Aggressive optimization (may be slower to compile)
};

/**
 * Compilation options
 */
struct CompileOptions {
    Target target;               // Target backend
    OptLevel opt_level = OptLevel::O2;
    bool enable_fast_math = true;
    bool enable_xla_fusion = true;
    bool debug_info = false;
    int64_t max_batch_size = 0;  // 0 = dynamic
};

/**
 * Compilation result
 */
struct CompileResult {
    bool success;
    void* executable;
    std::string error_message;
    int64_t compile_time_ms;
    size_t executable_size_bytes;
};

/**
 * XLACompiler - Compiles StableHLO to executable code
 *
 * Takes a StableHLO module and compiles it to native code
 * for the specified target backend.
 */
class XLACompiler {
public:
    XLACompiler();
    ~XLACompiler();

    // Non-copyable
    XLACompiler(const XLACompiler&) = delete;
    XLACompiler& operator=(const XLACompiler&) = delete;

    // ===== Compilation =====

    /**
     * Compile StableHLO module to executable.
     * @param module StableHLO module from emitter
     * @param options Compilation options
     * @return Compilation result
     */
    CompileResult compile(void* module, const CompileOptions& options);

    /**
     * Compile with default options for target.
     * @param module StableHLO module
     * @param target Target backend
     * @return Compilation result
     */
    CompileResult compileForTarget(void* module, Target target);

    // ===== Target Queries =====

    /**
     * Check if a target is available on this system.
     * @param target Target to check
     * @return true if available
     */
    static bool isTargetAvailable(Target target);

    /**
     * Get default target for this system.
     * @return Best available target
     */
    static Target getDefaultTarget();

    /**
     * Get all available targets.
     * @return Vector of available targets
     */
    static std::vector<Target> getAvailableTargets();

    // ===== Executable Management =====

    /**
     * Free a compiled executable.
     * @param executable Executable to free
     */
    void freeExecutable(void* executable);

    /**
     * Serialize executable to bytes for caching.
     * @param executable Executable to serialize
     * @return Serialized bytes (empty on failure)
     */
    std::vector<uint8_t> serializeExecutable(void* executable);

    /**
     * Deserialize executable from bytes.
     * @param data Serialized bytes
     * @param target Target the executable was compiled for
     * @return Deserialized executable (nullptr on failure)
     */
    void* deserializeExecutable(const std::vector<uint8_t>& data, Target target);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace xla
} // namespace eshkol

#endif // ESHKOL_XLA_COMPILER_H
