/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol Runtime Configuration Implementation
 *
 * Loads configuration from command-line, environment, and config files.
 */

#include <eshkol/core/config.h>
#include <eshkol/logger.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>
#else
#include <unistd.h>
#include <pwd.h>
#include <sys/stat.h>
#endif

namespace {

// ============================================================================
// Global State
// ============================================================================

std::mutex g_config_mutex;
eshkol_config_t g_config;
bool g_config_initialized = false;

// ============================================================================
// Helper Functions
// ============================================================================

// Get home directory path
std::string get_home_dir() {
#ifdef _WIN32
    char path[MAX_PATH];
    if (SUCCEEDED(SHGetFolderPathA(NULL, CSIDL_PROFILE, NULL, 0, path))) {
        return std::string(path);
    }
    return "";
#else
    const char* home = std::getenv("HOME");
    if (home) return std::string(home);

    struct passwd* pw = getpwuid(getuid());
    if (pw) return std::string(pw->pw_dir);

    return "";
#endif
}

// Check if file exists
bool file_exists(const std::string& path) {
#ifdef _WIN32
    DWORD attrs = GetFileAttributesA(path.c_str());
    return (attrs != INVALID_FILE_ATTRIBUTES && !(attrs & FILE_ATTRIBUTE_DIRECTORY));
#else
    struct stat st;
    return (stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode));
#endif
}

// Parse size with K/M/G suffix
size_t parse_size(const char* str) {
    if (!str || !*str) return 0;

    char* end = nullptr;
    double value = strtod(str, &end);
    if (end == str) return 0;

    if (end && *end) {
        switch (*end) {
            case 'K': case 'k': value *= 1024; break;
            case 'M': case 'm': value *= 1024 * 1024; break;
            case 'G': case 'g': value *= 1024 * 1024 * 1024; break;
        }
    }

    return static_cast<size_t>(value);
}

// Parse boolean
bool parse_bool(const char* str) {
    if (!str) return false;
    std::string s(str);
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return (s == "true" || s == "1" || s == "yes" || s == "on");
}

// Parse log level
eshkol_log_level_t parse_log_level(const char* str) {
    if (!str) return ESHKOL_LOG_LEVEL_INFO;
    std::string s(str);
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);

    if (s == "debug") return ESHKOL_LOG_LEVEL_DEBUG;
    if (s == "info") return ESHKOL_LOG_LEVEL_INFO;
    if (s == "warn" || s == "warning") return ESHKOL_LOG_LEVEL_WARN;
    if (s == "error") return ESHKOL_LOG_LEVEL_ERROR;
    if (s == "none" || s == "off") return ESHKOL_LOG_LEVEL_NONE;

    return ESHKOL_LOG_LEVEL_INFO;
}

// Parse log format
eshkol_log_format_t parse_log_format(const char* str) {
    if (!str) return ESHKOL_LOG_FORMAT_TEXT;
    std::string s(str);
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);

    if (s == "json") return ESHKOL_LOG_FORMAT_JSON;
    return ESHKOL_LOG_FORMAT_TEXT;
}

// Trim whitespace
std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// Remove quotes from string
std::string unquote(const std::string& s) {
    if (s.length() >= 2) {
        if ((s.front() == '"' && s.back() == '"') ||
            (s.front() == '\'' && s.back() == '\'')) {
            return s.substr(1, s.length() - 2);
        }
    }
    return s;
}

// ============================================================================
// Simple TOML-like Parser
// ============================================================================

struct ConfigSection {
    std::string name;
    std::vector<std::pair<std::string, std::string>> values;
};

std::vector<ConfigSection> parse_toml(const std::string& content) {
    std::vector<ConfigSection> sections;
    ConfigSection current;
    current.name = "";  // Global section

    std::istringstream stream(content);
    std::string line;

    while (std::getline(stream, line)) {
        line = trim(line);

        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;

        // Section header
        if (line[0] == '[' && line.back() == ']') {
            if (!current.name.empty() || !current.values.empty()) {
                sections.push_back(current);
            }
            current.name = trim(line.substr(1, line.length() - 2));
            current.values.clear();
            continue;
        }

        // Key-value pair
        size_t eq_pos = line.find('=');
        if (eq_pos != std::string::npos) {
            std::string key = trim(line.substr(0, eq_pos));
            std::string value = trim(line.substr(eq_pos + 1));
            value = unquote(value);
            current.values.push_back({key, value});
        }
    }

    // Don't forget the last section
    if (!current.name.empty() || !current.values.empty()) {
        sections.push_back(current);
    }

    return sections;
}

// Apply parsed config to struct
void apply_config_section(eshkol_config_t* config, const ConfigSection& section) {
    for (const auto& [key, value] : section.values) {
        // Runtime section
        if (section.name == "runtime" || section.name.empty()) {
            if (key == "max_heap") {
                config->max_heap_bytes = parse_size(value.c_str());
            } else if (key == "timeout_ms") {
                config->timeout_ms = static_cast<uint64_t>(atoll(value.c_str()));
            } else if (key == "max_stack") {
                config->max_stack_depth = static_cast<size_t>(atoll(value.c_str()));
            } else if (key == "max_tensor_elements") {
                config->max_tensor_elements = parse_size(value.c_str());
            } else if (key == "max_string_length") {
                config->max_string_length = parse_size(value.c_str());
            }
        }

        // Logging section
        if (section.name == "logging" || section.name.empty()) {
            if (key == "level" || key == "log_level") {
                config->log_level = parse_log_level(value.c_str());
            } else if (key == "format" || key == "log_format") {
                config->log_format = parse_log_format(value.c_str());
            } else if (key == "file" || key == "log_file") {
                // Allocate copy of string
                config->log_file = strdup(value.c_str());
            }
        }

        // Optimization section
        if (section.name == "optimization" || section.name.empty()) {
            if (key == "llvm_opt_level" || key == "opt_level") {
                int level = atoi(value.c_str());
                if (level >= 0 && level <= 3) {
                    config->llvm_opt_level = static_cast<eshkol_opt_level_t>(level);
                }
            } else if (key == "enable_simd") {
                config->enable_simd = parse_bool(value.c_str());
            } else if (key == "enable_xla") {
                config->enable_xla = parse_bool(value.c_str());
            } else if (key == "enable_gpu") {
                config->enable_gpu = parse_bool(value.c_str());
            }
        }

        // Debug section
        if (section.name == "debug" || section.name.empty()) {
            if (key == "dump_ast") {
                config->dump_ast = parse_bool(value.c_str());
            } else if (key == "dump_ir") {
                config->dump_ir = parse_bool(value.c_str());
            } else if (key == "debug_mode" || key == "debug") {
                config->debug_mode = parse_bool(value.c_str());
            }
        }

        // Features section
        if (section.name == "features" || section.name.empty()) {
            if (key == "strict_mode") {
                config->strict_mode = parse_bool(value.c_str());
            } else if (key == "enable_warnings") {
                config->enable_warnings = parse_bool(value.c_str());
            } else if (key == "color_output") {
                config->color_output = parse_bool(value.c_str());
            }
        }
    }
}

} // anonymous namespace

// ============================================================================
// C API Implementation
// ============================================================================

extern "C" {

eshkol_config_t eshkol_config_defaults(void) {
    eshkol_config_t config = {};

    // Runtime defaults
    config.max_heap_bytes = 1ULL * 1024 * 1024 * 1024;  // 1GB
    config.timeout_ms = 0;  // No timeout by default
    config.max_stack_depth = 10000;
    config.max_tensor_elements = 1ULL * 1000 * 1000 * 1000;  // 1 billion
    config.max_string_length = 100ULL * 1024 * 1024;  // 100MB

    // Logging defaults
    config.log_level = ESHKOL_LOG_LEVEL_INFO;
    config.log_format = ESHKOL_LOG_FORMAT_TEXT;
    config.log_file = nullptr;

    // Optimization defaults
    config.llvm_opt_level = ESHKOL_OPT_LEVEL_2;
    config.enable_simd = true;
    config.enable_xla = false;
    config.enable_gpu = false;

    // Debug defaults
    config.dump_ast = false;
    config.dump_ir = false;
    config.debug_mode = false;

    // Library paths
    config.lib_paths = nullptr;
    config.lib_path_count = 0;

    // Feature defaults
    config.strict_mode = false;
    config.enable_warnings = true;
    config.color_output = true;

    return config;
}

int eshkol_config_load(eshkol_config_t* config) {
    if (!config) return -1;

    // Start with defaults
    *config = eshkol_config_defaults();

    // Load from config file (lowest priority)
    eshkol_config_load_file(config, nullptr);

    // Load from environment (medium priority)
    eshkol_config_load_env(config);

    return 0;
}

int eshkol_config_load_file(eshkol_config_t* config, const char* path) {
    if (!config) return -1;

    std::vector<std::string> search_paths;

    if (path) {
        search_paths.push_back(path);
    } else {
        // Search default locations
        search_paths.push_back(".eshkol.toml");
        search_paths.push_back("eshkol.toml");

        std::string home = get_home_dir();
        if (!home.empty()) {
            search_paths.push_back(home + "/.config/eshkol/config.toml");
            search_paths.push_back(home + "/.eshkol/config.toml");
        }
    }

    for (const auto& search_path : search_paths) {
        if (file_exists(search_path)) {
            std::ifstream file(search_path);
            if (file.is_open()) {
                std::stringstream buffer;
                buffer << file.rdbuf();
                std::string content = buffer.str();
                file.close();

                auto sections = parse_toml(content);
                for (const auto& section : sections) {
                    apply_config_section(config, section);
                }

                eshkol_debug("Loaded config from: %s", search_path.c_str());
                return 0;
            }
        }
    }

    return 0;  // No config file found is not an error
}

void eshkol_config_load_env(eshkol_config_t* config) {
    if (!config) return;

    // Runtime
    const char* val;

    if ((val = std::getenv(ESHKOL_ENV_MAX_HEAP))) {
        config->max_heap_bytes = parse_size(val);
    }
    if ((val = std::getenv(ESHKOL_ENV_TIMEOUT_MS))) {
        config->timeout_ms = static_cast<uint64_t>(atoll(val));
    }
    if ((val = std::getenv(ESHKOL_ENV_MAX_STACK))) {
        config->max_stack_depth = static_cast<size_t>(atoll(val));
    }
    if ((val = std::getenv(ESHKOL_ENV_MAX_TENSOR))) {
        config->max_tensor_elements = parse_size(val);
    }
    if ((val = std::getenv(ESHKOL_ENV_MAX_STRING))) {
        config->max_string_length = parse_size(val);
    }

    // Logging
    if ((val = std::getenv(ESHKOL_ENV_LOG_LEVEL))) {
        config->log_level = parse_log_level(val);
    }
    if ((val = std::getenv(ESHKOL_ENV_LOG_FORMAT))) {
        config->log_format = parse_log_format(val);
    }
    if ((val = std::getenv(ESHKOL_ENV_LOG_FILE))) {
        config->log_file = strdup(val);
    }

    // Optimization
    if ((val = std::getenv(ESHKOL_ENV_OPT_LEVEL))) {
        int level = atoi(val);
        if (level >= 0 && level <= 3) {
            config->llvm_opt_level = static_cast<eshkol_opt_level_t>(level);
        }
    }
    if ((val = std::getenv(ESHKOL_ENV_ENABLE_SIMD))) {
        config->enable_simd = parse_bool(val);
    }
    if ((val = std::getenv(ESHKOL_ENV_ENABLE_XLA))) {
        config->enable_xla = parse_bool(val);
    }
    if ((val = std::getenv(ESHKOL_ENV_ENABLE_GPU))) {
        config->enable_gpu = parse_bool(val);
    }

    // Debug
    if ((val = std::getenv(ESHKOL_ENV_DEBUG))) {
        config->debug_mode = parse_bool(val);
    }

    // Library paths (colon-separated)
    if ((val = std::getenv(ESHKOL_ENV_LIB_PATH))) {
        std::string paths(val);
        std::vector<std::string> path_vec;
        std::istringstream stream(paths);
        std::string path;

#ifdef _WIN32
        char delim = ';';
#else
        char delim = ':';
#endif

        while (std::getline(stream, path, delim)) {
            if (!path.empty()) {
                path_vec.push_back(path);
            }
        }

        if (!path_vec.empty()) {
            config->lib_path_count = path_vec.size();
            config->lib_paths = new const char*[path_vec.size()];
            for (size_t i = 0; i < path_vec.size(); i++) {
                config->lib_paths[i] = strdup(path_vec[i].c_str());
            }
        }
    }
}

void eshkol_config_apply_args(eshkol_config_t* config, int argc, char** argv) {
    if (!config || argc <= 0 || !argv) return;

    // Command-line parsing is handled in eshkol-run.cpp
    // This function is for future extensibility
}

const eshkol_config_t* eshkol_config_get(void) {
    std::lock_guard<std::mutex> lock(g_config_mutex);

    if (!g_config_initialized) {
        eshkol_config_load(&g_config);
        g_config_initialized = true;
    }

    return &g_config;
}

void eshkol_config_set(const eshkol_config_t* config) {
    if (!config) return;

    std::lock_guard<std::mutex> lock(g_config_mutex);
    g_config = *config;
    g_config_initialized = true;
}

// Individual getters
size_t eshkol_config_get_max_heap(void) {
    return eshkol_config_get()->max_heap_bytes;
}

uint64_t eshkol_config_get_timeout(void) {
    return eshkol_config_get()->timeout_ms;
}

size_t eshkol_config_get_max_stack(void) {
    return eshkol_config_get()->max_stack_depth;
}

eshkol_log_level_t eshkol_config_get_log_level(void) {
    return eshkol_config_get()->log_level;
}

eshkol_log_format_t eshkol_config_get_log_format(void) {
    return eshkol_config_get()->log_format;
}

eshkol_opt_level_t eshkol_config_get_opt_level(void) {
    return eshkol_config_get()->llvm_opt_level;
}

bool eshkol_config_is_simd_enabled(void) {
    return eshkol_config_get()->enable_simd;
}

bool eshkol_config_is_xla_enabled(void) {
    return eshkol_config_get()->enable_xla;
}

bool eshkol_config_is_gpu_enabled(void) {
    return eshkol_config_get()->enable_gpu;
}

bool eshkol_config_is_debug_mode(void) {
    return eshkol_config_get()->debug_mode;
}

void eshkol_config_cleanup(eshkol_config_t* config) {
    if (!config) return;

    if (config->log_file) {
        free(const_cast<char*>(config->log_file));
        config->log_file = nullptr;
    }

    if (config->lib_paths) {
        for (size_t i = 0; i < config->lib_path_count; i++) {
            free(const_cast<char*>(config->lib_paths[i]));
        }
        delete[] config->lib_paths;
        config->lib_paths = nullptr;
        config->lib_path_count = 0;
    }
}

} // extern "C"

// ============================================================================
// C++ Implementation
// ============================================================================

namespace eshkol {

Config& Config::instance() {
    static Config instance;
    return instance;
}

Config::Config() {
    config_ = eshkol_config_defaults();
}

Config::~Config() {
    eshkol_config_cleanup(&config_);
}

bool Config::load() {
    if (loaded_) return true;

    int result = eshkol_config_load(&config_);
    loaded_ = (result == 0);
    return loaded_;
}

std::vector<std::string> Config::libPaths() const {
    std::vector<std::string> paths;
    for (size_t i = 0; i < config_.lib_path_count; i++) {
        if (config_.lib_paths[i]) {
            paths.push_back(config_.lib_paths[i]);
        }
    }
    return paths;
}

} // namespace eshkol
