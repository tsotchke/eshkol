/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol Runtime Configuration
 *
 * Provides unified configuration from multiple sources:
 * 1. Command-line flags (highest priority)
 * 2. Environment variables
 * 3. Config file (~/.eshkol/config.toml or .eshkol.toml)
 * 4. Default values (lowest priority)
 */
#ifndef ESHKOL_CORE_CONFIG_H
#define ESHKOL_CORE_CONFIG_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Configuration Structure
// ============================================================================

typedef enum {
    ESHKOL_LOG_LEVEL_DEBUG = 0,
    ESHKOL_LOG_LEVEL_INFO,
    ESHKOL_LOG_LEVEL_WARN,
    ESHKOL_LOG_LEVEL_ERROR,
    ESHKOL_LOG_LEVEL_NONE
} eshkol_log_level_t;

typedef enum {
    ESHKOL_LOG_FORMAT_TEXT = 0,  // Human-readable text
    ESHKOL_LOG_FORMAT_JSON       // Structured JSON
} eshkol_log_format_t;

typedef enum {
    ESHKOL_OPT_LEVEL_0 = 0,  // No optimization
    ESHKOL_OPT_LEVEL_1,      // Basic optimization
    ESHKOL_OPT_LEVEL_2,      // Standard optimization (default)
    ESHKOL_OPT_LEVEL_3       // Aggressive optimization
} eshkol_opt_level_t;

typedef struct eshkol_config {
    // Runtime settings
    size_t max_heap_bytes;           // Maximum heap allocation
    uint64_t timeout_ms;             // Execution timeout (0 = unlimited)
    size_t max_stack_depth;          // Maximum recursion depth
    size_t max_tensor_elements;      // Maximum tensor elements
    size_t max_string_length;        // Maximum string length

    // Logging settings
    eshkol_log_level_t log_level;    // Minimum log level
    eshkol_log_format_t log_format;  // Output format (text/json)
    const char* log_file;            // Log file path (NULL = stderr)

    // Optimization settings
    eshkol_opt_level_t llvm_opt_level;  // LLVM optimization level
    bool enable_simd;                    // Enable SIMD vectorization
    bool enable_xla;                     // Enable XLA backend
    bool enable_gpu;                     // Enable GPU acceleration

    // Debug settings
    bool dump_ast;                   // Dump AST to stderr
    bool dump_ir;                    // Dump LLVM IR to stderr
    bool debug_mode;                 // Enable debug output

    // Library paths
    const char** lib_paths;          // Library search paths
    size_t lib_path_count;           // Number of library paths

    // Feature flags
    bool strict_mode;                // Strict R7RS compliance
    bool enable_warnings;            // Enable warnings
    bool color_output;               // Colorized output

    // Type system flags
    bool strict_types;               // Strict type checking (errors instead of warnings)
    bool unsafe_mode;                // Skip all type checks
} eshkol_config_t;

// ============================================================================
// Configuration Source Priority
// ============================================================================

typedef enum {
    ESHKOL_CONFIG_SRC_DEFAULT = 0,   // Built-in defaults
    ESHKOL_CONFIG_SRC_FILE,          // Config file
    ESHKOL_CONFIG_SRC_ENV,           // Environment variable
    ESHKOL_CONFIG_SRC_CMDLINE        // Command-line flag
} eshkol_config_source_t;

// ============================================================================
// Initialization and Loading
// ============================================================================

// Get default configuration
eshkol_config_t eshkol_config_defaults(void);

// Load configuration from all sources (in priority order)
// Returns: 0 on success, non-zero on error
int eshkol_config_load(eshkol_config_t* config);

// Load configuration from file
// - path: Path to config file (NULL = search default locations)
// Returns: 0 on success, non-zero on error (file not found is not an error)
int eshkol_config_load_file(eshkol_config_t* config, const char* path);

// Load configuration from environment variables
void eshkol_config_load_env(eshkol_config_t* config);

// Apply command-line arguments to config
// - argc/argv: Command-line arguments
// - config: Configuration to update
void eshkol_config_apply_args(eshkol_config_t* config, int argc, char** argv);

// ============================================================================
// Accessors
// ============================================================================

// Get the global configuration (initialized on first call)
const eshkol_config_t* eshkol_config_get(void);

// Set the global configuration
void eshkol_config_set(const eshkol_config_t* config);

// ============================================================================
// Individual Setting Getters
// ============================================================================

size_t eshkol_config_get_max_heap(void);
uint64_t eshkol_config_get_timeout(void);
size_t eshkol_config_get_max_stack(void);
eshkol_log_level_t eshkol_config_get_log_level(void);
eshkol_log_format_t eshkol_config_get_log_format(void);
eshkol_opt_level_t eshkol_config_get_opt_level(void);
bool eshkol_config_is_simd_enabled(void);
bool eshkol_config_is_xla_enabled(void);
bool eshkol_config_is_gpu_enabled(void);
bool eshkol_config_is_debug_mode(void);

// ============================================================================
// Environment Variable Names
// ============================================================================

// Runtime
#define ESHKOL_ENV_MAX_HEAP         "ESHKOL_MAX_HEAP"
#define ESHKOL_ENV_TIMEOUT_MS       "ESHKOL_TIMEOUT_MS"
#define ESHKOL_ENV_MAX_STACK        "ESHKOL_MAX_STACK"
#define ESHKOL_ENV_MAX_TENSOR       "ESHKOL_MAX_TENSOR_ELEMS"
#define ESHKOL_ENV_MAX_STRING       "ESHKOL_MAX_STRING_LEN"

// Logging
#define ESHKOL_ENV_LOG_LEVEL        "ESHKOL_LOG_LEVEL"
#define ESHKOL_ENV_LOG_FORMAT       "ESHKOL_LOG_FORMAT"
#define ESHKOL_ENV_LOG_FILE         "ESHKOL_LOG_FILE"

// Optimization
#define ESHKOL_ENV_OPT_LEVEL        "ESHKOL_OPT_LEVEL"
#define ESHKOL_ENV_ENABLE_SIMD      "ESHKOL_ENABLE_SIMD"
#define ESHKOL_ENV_ENABLE_XLA       "ESHKOL_ENABLE_XLA"
#define ESHKOL_ENV_ENABLE_GPU       "ESHKOL_ENABLE_GPU"

// Debug
#define ESHKOL_ENV_DEBUG            "ESHKOL_DEBUG"

// Library paths (colon-separated)
#define ESHKOL_ENV_LIB_PATH         "ESHKOL_LIB_PATH"

// ============================================================================
// Config File Locations (searched in order)
// ============================================================================

// 1. ./.eshkol.toml (project-local)
// 2. ~/.config/eshkol/config.toml (XDG config)
// 3. ~/.eshkol/config.toml (home directory)

// ============================================================================
// Cleanup
// ============================================================================

// Free any allocated memory in config
void eshkol_config_cleanup(eshkol_config_t* config);

#ifdef __cplusplus
}
#endif

// ============================================================================
// C++ Helpers
// ============================================================================

#ifdef __cplusplus

#include <string>
#include <vector>

namespace eshkol {

// C++ wrapper for configuration
class Config {
public:
    static Config& instance();

    // Load from all sources
    bool load();

    // Accessors
    size_t maxHeap() const { return config_.max_heap_bytes; }
    uint64_t timeout() const { return config_.timeout_ms; }
    size_t maxStack() const { return config_.max_stack_depth; }

    eshkol_log_level_t logLevel() const { return config_.log_level; }
    eshkol_log_format_t logFormat() const { return config_.log_format; }
    std::string logFile() const { return config_.log_file ? config_.log_file : ""; }

    eshkol_opt_level_t optLevel() const { return config_.llvm_opt_level; }
    bool simdEnabled() const { return config_.enable_simd; }
    bool xlaEnabled() const { return config_.enable_xla; }
    bool gpuEnabled() const { return config_.enable_gpu; }

    bool debugMode() const { return config_.debug_mode; }
    bool dumpAst() const { return config_.dump_ast; }
    bool dumpIr() const { return config_.dump_ir; }

    bool strictTypes() const { return config_.strict_types; }
    bool unsafeMode() const { return config_.unsafe_mode; }

    std::vector<std::string> libPaths() const;

    // Raw access
    const eshkol_config_t& raw() const { return config_; }
    eshkol_config_t& raw() { return config_; }

private:
    Config();
    ~Config();

    eshkol_config_t config_;
    bool loaded_ = false;
};

} // namespace eshkol

#endif // __cplusplus

#endif // ESHKOL_CORE_CONFIG_H
