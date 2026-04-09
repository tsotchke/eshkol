/*
 * Eshkol Package Manager (eshkol-pkg)
 *
 * CLI tool for managing Eshkol project dependencies and builds.
 * Uses eshkol.toml manifest format and git-based package registry.
 *
 * Commands:
 *   init       Create a new eshkol.toml manifest
 *   build      Compile the current project
 *   install    Install dependencies from eshkol.toml
 *   add        Add a dependency to eshkol.toml
 *   remove     Remove a dependency from eshkol.toml
 *   search     Search the package registry
 *   publish    Publish a package to the registry
 *   run        Build and run the project
 *   clean      Remove build artifacts
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#if defined(__has_include)
#if __has_include(<filesystem>)
#include <filesystem>
#define ESHKOL_PKG_HAVE_STD_FILESYSTEM 1
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
#define ESHKOL_PKG_HAVE_EXPERIMENTAL_FILESYSTEM 1
#endif
#else
#include <filesystem>
#define ESHKOL_PKG_HAVE_STD_FILESYSTEM 1
#endif
#include <functional>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#if defined(__has_include)
#if __has_include(<windows.h>)
#include <windows.h>
#endif
#else
#include <windows.h>
#endif
#else
#include <cerrno>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#if defined(ESHKOL_PKG_HAVE_STD_FILESYSTEM)
namespace fs = std::filesystem;
#else
namespace fs = std::experimental::filesystem;
#endif

// ============================================================================
// Minimal TOML Parser (supports tables, key=value, strings, arrays)
// ============================================================================

struct TomlValue {
    enum Type { String, Integer, Float, Bool, Array, Table };
    Type type = String;
    std::string str_val;
    int64_t int_val = 0;
    double float_val = 0;
    bool bool_val = false;
    std::vector<TomlValue> arr_val;
    std::map<std::string, TomlValue> table_val;

    TomlValue() : type(String) {}
    explicit TomlValue(const std::string& s) : type(String), str_val(s) {}
    explicit TomlValue(int64_t i) : type(Integer), int_val(i) {}
    explicit TomlValue(double f) : type(Float), float_val(f) {}
    explicit TomlValue(bool b) : type(Bool), bool_val(b) {}

    static TomlValue make_table() { TomlValue v; v.type = Table; return v; }
    static TomlValue make_array() { TomlValue v; v.type = Array; return v; }
};

class TomlParser {
public:
    static std::unordered_map<std::string, TomlValue> parse(const std::string& content) {
        std::unordered_map<std::string, TomlValue> root;
        std::string current_table;
        std::istringstream stream(content);
        std::string line;

        while (std::getline(stream, line)) {
            // Strip comments
            size_t comment_pos = line.find('#');
            if (comment_pos != std::string::npos) {
                // Don't strip inside strings
                bool in_string = false;
                for (size_t i = 0; i < comment_pos; i++) {
                    if (line[i] == '"') in_string = !in_string;
                }
                if (!in_string) line = line.substr(0, comment_pos);
            }

            // Trim whitespace
            line = trim(line);
            if (line.empty()) continue;

            // Table header [name]
            if (line.front() == '[' && line.back() == ']') {
                current_table = line.substr(1, line.size() - 2);
                current_table = trim(current_table);
                root[current_table] = TomlValue::make_table();
                continue;
            }

            // Key = value
            size_t eq_pos = line.find('=');
            if (eq_pos != std::string::npos) {
                std::string key = trim(line.substr(0, eq_pos));
                std::string val_str = trim(line.substr(eq_pos + 1));
                TomlValue value = parse_value(val_str);

                if (current_table.empty()) {
                    root[key] = value;
                } else {
                    root[current_table].table_val[key] = value;
                }
            }
        }
        return root;
    }

    static std::string serialize(const std::unordered_map<std::string, TomlValue>& data) {
        std::ostringstream ss;

        // Write top-level keys first
        for (const auto& [key, val] : data) {
            if (val.type != TomlValue::Table) {
                ss << key << " = ";
                serialize_value(ss, val);
                ss << "\n";
            }
        }

        // Write tables
        for (const auto& [key, val] : data) {
            if (val.type == TomlValue::Table) {
                ss << "\n[" << key << "]\n";
                for (const auto& [k, v] : val.table_val) {
                    ss << k << " = ";
                    serialize_value(ss, v);
                    ss << "\n";
                }
            }
        }
        return ss.str();
    }

private:
    static std::string trim(const std::string& s) {
        size_t start = s.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) return "";
        size_t end = s.find_last_not_of(" \t\r\n");
        return s.substr(start, end - start + 1);
    }

    static TomlValue parse_value(const std::string& s) {
        if (s.empty()) return TomlValue("");

        // String
        if (s.front() == '"') {
            size_t end = s.rfind('"');
            if (end > 0) return TomlValue(s.substr(1, end - 1));
            return TomlValue(s.substr(1));
        }

        // Boolean
        if (s == "true") return TomlValue(true);
        if (s == "false") return TomlValue(false);

        // Array
        if (s.front() == '[') {
            TomlValue arr = TomlValue::make_array();
            std::string inner = s.substr(1, s.size() - 2);
            // Simple comma-separated array
            size_t pos = 0;
            while (pos < inner.size()) {
                // Skip whitespace
                while (pos < inner.size() && (inner[pos] == ' ' || inner[pos] == '\t'))
                    pos++;
                if (pos >= inner.size()) break;

                // Find end of value (comma or end)
                size_t end = inner.find(',', pos);
                if (end == std::string::npos) end = inner.size();
                std::string val_str = trim(inner.substr(pos, end - pos));
                if (!val_str.empty()) {
                    arr.arr_val.push_back(parse_value(val_str));
                }
                pos = end + 1;
            }
            return arr;
        }

        // Float
        if (s.find('.') != std::string::npos) {
            try { return TomlValue(std::stod(s)); }
            catch (...) { return TomlValue(s); }
        }

        // Integer
        try { return TomlValue(static_cast<int64_t>(std::stoll(s))); }
        catch (...) { return TomlValue(s); }
    }

    static void serialize_value(std::ostringstream& ss, const TomlValue& val) {
        switch (val.type) {
            case TomlValue::String:
                ss << "\"" << val.str_val << "\"";
                break;
            case TomlValue::Integer:
                ss << val.int_val;
                break;
            case TomlValue::Float:
                ss << val.float_val;
                break;
            case TomlValue::Bool:
                ss << (val.bool_val ? "true" : "false");
                break;
            case TomlValue::Array: {
                ss << "[";
                for (size_t i = 0; i < val.arr_val.size(); i++) {
                    if (i > 0) ss << ", ";
                    serialize_value(ss, val.arr_val[i]);
                }
                ss << "]";
                break;
            }
            case TomlValue::Table: {
                ss << "{";
                bool first = true;
                for (const auto& [k, v] : val.table_val) {
                    if (!first) ss << ", ";
                    first = false;
                    ss << k << " = ";
                    serialize_value(ss, v);
                }
                ss << "}";
                break;
            }
        }
    }
};

// ============================================================================
// Manifest (eshkol.toml)
// ============================================================================

struct Dependency {
    std::string name;
    std::string version;  // semver or git URL
    std::string source;   // "registry" or "git"
};

struct Manifest {
    std::string name;
    std::string version;
    std::string description;
    std::string author;
    std::string license;
    std::string entry;    // main .esk file
    std::vector<std::string> sources;
    std::vector<Dependency> dependencies;
};

Manifest load_manifest(const fs::path& dir) {
    Manifest m;
    fs::path manifest_path = dir / "eshkol.toml";

    if (!fs::exists(manifest_path)) {
        std::cerr << "Error: eshkol.toml not found in " << dir << std::endl;
        std::exit(1);
    }

    std::ifstream file(manifest_path);
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    file.close();

    auto data = TomlParser::parse(content);

    // [package] section
    if (data.count("package")) {
        auto& pkg = data["package"].table_val;
        if (pkg.count("name")) m.name = pkg["name"].str_val;
        if (pkg.count("version")) m.version = pkg["version"].str_val;
        if (pkg.count("description")) m.description = pkg["description"].str_val;
        if (pkg.count("author")) m.author = pkg["author"].str_val;
        if (pkg.count("license")) m.license = pkg["license"].str_val;
        if (pkg.count("entry")) m.entry = pkg["entry"].str_val;
        if (pkg.count("sources")) {
            for (auto& s : pkg["sources"].arr_val)
                m.sources.push_back(s.str_val);
        }
    }

    // [dependencies] section
    if (data.count("dependencies")) {
        for (auto& [name, val] : data["dependencies"].table_val) {
            Dependency dep;
            dep.name = name;
            dep.version = val.str_val;
            dep.source = "registry";
            m.dependencies.push_back(dep);
        }
    }

    return m;
}

void save_manifest(const fs::path& dir, const Manifest& m) {
    std::unordered_map<std::string, TomlValue> data;

    // [package]
    TomlValue pkg = TomlValue::make_table();
    pkg.table_val["name"] = TomlValue(m.name);
    pkg.table_val["version"] = TomlValue(m.version);
    pkg.table_val["description"] = TomlValue(m.description);
    pkg.table_val["author"] = TomlValue(m.author);
    pkg.table_val["license"] = TomlValue(m.license);
    if (!m.entry.empty()) {
        pkg.table_val["entry"] = TomlValue(m.entry);
    }
    if (!m.sources.empty()) {
        TomlValue sources = TomlValue::make_array();
        for (auto& s : m.sources) sources.arr_val.push_back(TomlValue(s));
        pkg.table_val["sources"] = sources;
    }
    data["package"] = pkg;

    // [dependencies]
    if (!m.dependencies.empty()) {
        TomlValue deps = TomlValue::make_table();
        for (auto& dep : m.dependencies) {
            deps.table_val[dep.name] = TomlValue(dep.version);
        }
        data["dependencies"] = deps;
    }

    fs::path manifest_path = dir / "eshkol.toml";
    std::ofstream file(manifest_path);
    file << TomlParser::serialize(data);
    file.close();
}

// ============================================================================
// Package Registry (git-based)
// ============================================================================

static const char* DEFAULT_REGISTRY = "https://github.com/tsotchke/eshkol-registry.git";

std::string get_registry_url() {
    const char* env = std::getenv("ESHKOL_REGISTRY");
    return env ? env : DEFAULT_REGISTRY;
}

fs::path get_cache_dir() {
    const char* home = std::getenv("HOME");
    if (!home) home = std::getenv("USERPROFILE");
    if (!home) {
        std::cerr << "Error: Cannot determine home directory" << std::endl;
        std::exit(1);
    }
    fs::path cache = fs::path(home) / ".eshkol" / "cache";
    fs::create_directories(cache);
    return cache;
}

fs::path get_packages_dir() {
    fs::path pkgs = get_cache_dir() / "packages";
    fs::create_directories(pkgs);
    return pkgs;
}

namespace {

#ifdef _WIN32
std::wstring widen_utf8(const std::string& text) {
    if (text.empty()) return {};
    int size = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, nullptr, 0);
    if (size <= 0) return {};
    std::wstring wide(static_cast<size_t>(size - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, wide.data(), size);
    return wide;
}

std::wstring build_windows_command_line(const std::vector<std::string>& args) {
    std::wstring command_line;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0) command_line.push_back(L' ');
        command_line.push_back(L'"');
        const std::wstring arg = widen_utf8(args[i]);
        for (wchar_t ch : arg) {
            if (ch == L'\\' || ch == L'"') {
                command_line.push_back(L'\\');
            }
            command_line.push_back(ch);
        }
        command_line.push_back(L'"');
    }
    return command_line;
}
#endif

} // namespace

int run_command(const std::vector<std::string>& args) {
    if (args.empty()) {
        return 1;
    }

#ifdef _WIN32
    const std::wstring application = widen_utf8(args.front());
    std::wstring command_line = build_windows_command_line(args);
    std::vector<wchar_t> mutable_command_line(command_line.begin(), command_line.end());
    mutable_command_line.push_back(L'\0');

    STARTUPINFOW startup_info{};
    startup_info.cb = sizeof(startup_info);

    PROCESS_INFORMATION process_info{};
    if (!CreateProcessW(application.c_str(),
                        mutable_command_line.data(),
                        nullptr,
                        nullptr,
                        FALSE,
                        0,
                        nullptr,
                        nullptr,
                        &startup_info,
                        &process_info)) {
        return static_cast<int>(GetLastError());
    }

    WaitForSingleObject(process_info.hProcess, INFINITE);

    DWORD exit_code = 1;
    GetExitCodeProcess(process_info.hProcess, &exit_code);
    CloseHandle(process_info.hThread);
    CloseHandle(process_info.hProcess);
    return static_cast<int>(exit_code);
#else
    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (const auto& arg : args) {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }
    argv.push_back(nullptr);

    pid_t pid = fork();
    if (pid == 0) {
        execvp(argv[0], argv.data());
        _exit(errno == ENOENT ? 127 : 126);
    }
    if (pid < 0) {
        return errno;
    }

    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            return errno;
        }
    }

    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        return 128 + WTERMSIG(status);
    }
    return 1;
#endif
}

// ============================================================================
// Commands
// ============================================================================

int cmd_init(int argc, char* argv[]) {
    fs::path dir = fs::current_path();
    fs::path manifest_path = dir / "eshkol.toml";

    if (fs::exists(manifest_path)) {
        std::cerr << "Error: eshkol.toml already exists in this directory" << std::endl;
        return 1;
    }

    // Derive project name from directory
    std::string project_name = dir.filename().string();

    Manifest m;
    m.name = project_name;
    m.version = "0.1.0";
    m.description = "";
    m.author = "";
    m.license = "MIT";
    m.entry = "src/main.esk";
    m.sources = {"src/*.esk"};

    save_manifest(dir, m);

    // Create src directory and main.esk template
    fs::create_directories(dir / "src");

    fs::path main_esk = dir / "src" / "main.esk";
    if (!fs::exists(main_esk)) {
        std::ofstream f(main_esk);
        f << ";; " << project_name << " - main entry point\n";
        f << ";; Build: eshkol-pkg build\n";
        f << ";; Run:   eshkol-pkg run\n\n";
        f << "(display \"Hello from " << project_name << "!\")\n";
        f << "(newline)\n";
        f.close();
    }

    std::cout << "Created eshkol.toml for project '" << project_name << "'" << std::endl;
    std::cout << "  src/main.esk created" << std::endl;
    std::cout << "\nRun 'eshkol-pkg build' to compile." << std::endl;
    return 0;
}

int cmd_build(int argc, char* argv[]) {
    fs::path dir = fs::current_path();
    Manifest m = load_manifest(dir);

    std::string compiler = "eshkol-run";
    const char* env = std::getenv("ESHKOL_COMPILER");
    if (env) compiler = env;

    // Determine entry file
    std::string entry = m.entry.empty() ? "src/main.esk" : m.entry;
    fs::path entry_path = dir / entry;

    if (!fs::exists(entry_path)) {
        std::cerr << "Error: Entry file not found: " << entry_path << std::endl;
        return 1;
    }

    // Create build directory
    fs::path build_dir = dir / "build";
    fs::create_directories(build_dir);

    // Output binary name
    fs::path output = build_dir / m.name;

    // Build command
    std::vector<std::string> cmd = {
        compiler,
        entry_path.string(),
        "-o",
        output.string(),
    };

    // Add dependency library paths
    fs::path deps_dir = dir / "eshkol_deps";
    if (fs::exists(deps_dir)) {
        for (auto& dep_entry : fs::directory_iterator(deps_dir)) {
            if (dep_entry.is_directory()) {
                // Look for compiled .o files
                for (auto& obj : fs::directory_iterator(dep_entry.path())) {
                    if (obj.path().extension() == ".o") {
                        cmd.push_back("--link");
                        cmd.push_back(obj.path().string());
                    }
                }
            }
        }
    }

    std::cout << "Building " << m.name << " v" << m.version << "..." << std::endl;
    int result = run_command(cmd);

    if (result == 0) {
        std::cout << "Built: " << output << std::endl;
    } else {
        std::cerr << "Build failed with exit code " << result << std::endl;
    }
    return result;
}

int cmd_run(int argc, char* argv[]) {
    int result = cmd_build(argc, argv);
    if (result != 0) return result;

    fs::path dir = fs::current_path();
    Manifest m = load_manifest(dir);
    fs::path output = dir / "build" / m.name;

    std::cout << "\n--- Running " << m.name << " ---\n" << std::endl;
    return run_command({output.string()});
}

int cmd_install(int argc, char* argv[]) {
    fs::path dir = fs::current_path();
    Manifest m = load_manifest(dir);

    if (m.dependencies.empty()) {
        std::cout << "No dependencies to install." << std::endl;
        return 0;
    }

    fs::path deps_dir = dir / "eshkol_deps";
    fs::create_directories(deps_dir);
    fs::path pkgs = get_packages_dir();

    for (auto& dep : m.dependencies) {
        std::cout << "Installing " << dep.name << " " << dep.version << "..." << std::endl;

        // Check cache
        fs::path cached = pkgs / dep.name;
        if (!fs::exists(cached)) {
            // Clone from registry
            std::string registry = get_registry_url();
            std::string repo_url = registry;
            // For git-based registry, packages are subdirectories or separate repos
            // Convention: registry/<package-name>.git
            if (dep.source == "git" && !dep.version.empty() && dep.version.find("://") != std::string::npos) {
                repo_url = dep.version; // Direct git URL
            } else {
                // Try the package registry
                repo_url = "https://github.com/tsotchke/" + dep.name + ".git";
            }

            std::vector<std::string> cmd = {
                "git", "clone", "--depth", "1", repo_url, cached.string()
            };
            if (run_command(cmd) != 0) {
                std::cerr << "Warning: Could not fetch " << dep.name << " from " << repo_url << std::endl;
                continue;
            }
        }

        // Copy/link to local deps
        fs::path local_dep = deps_dir / dep.name;
        if (!fs::exists(local_dep)) {
            fs::create_directory_symlink(cached, local_dep);
        }

        std::cout << "  Installed " << dep.name << std::endl;
    }

    std::cout << "All dependencies installed." << std::endl;
    return 0;
}

int cmd_add(int argc, char* argv[]) {
    if (argc < 1) {
        std::cerr << "Usage: eshkol-pkg add <package> [version]" << std::endl;
        return 1;
    }

    std::string pkg_name = argv[0];
    std::string version = argc > 1 ? argv[1] : "*";

    fs::path dir = fs::current_path();
    Manifest m = load_manifest(dir);

    // Check if already exists
    for (auto& dep : m.dependencies) {
        if (dep.name == pkg_name) {
            dep.version = version;
            save_manifest(dir, m);
            std::cout << "Updated " << pkg_name << " to " << version << std::endl;
            return 0;
        }
    }

    Dependency dep;
    dep.name = pkg_name;
    dep.version = version;
    dep.source = "registry";
    m.dependencies.push_back(dep);
    save_manifest(dir, m);

    std::cout << "Added " << pkg_name << " " << version << " to dependencies" << std::endl;
    return 0;
}

int cmd_remove(int argc, char* argv[]) {
    if (argc < 1) {
        std::cerr << "Usage: eshkol-pkg remove <package>" << std::endl;
        return 1;
    }

    std::string pkg_name = argv[0];

    fs::path dir = fs::current_path();
    Manifest m = load_manifest(dir);

    auto it = std::remove_if(m.dependencies.begin(), m.dependencies.end(),
        [&](const Dependency& d) { return d.name == pkg_name; });

    if (it == m.dependencies.end()) {
        std::cerr << "Package " << pkg_name << " not found in dependencies" << std::endl;
        return 1;
    }

    m.dependencies.erase(it, m.dependencies.end());
    save_manifest(dir, m);

    // Remove local copy
    fs::path local_dep = dir / "eshkol_deps" / pkg_name;
    if (fs::exists(local_dep)) {
        fs::remove_all(local_dep);
    }

    std::cout << "Removed " << pkg_name << " from dependencies" << std::endl;
    return 0;
}

int cmd_search(int argc, char* argv[]) {
    if (argc < 1) {
        std::cerr << "Usage: eshkol-pkg search <query>" << std::endl;
        return 1;
    }

    std::string query = argv[0];

    // For now, search the local cache
    fs::path pkgs = get_packages_dir();
    bool found = false;

    if (fs::exists(pkgs)) {
        for (auto& entry : fs::directory_iterator(pkgs)) {
            if (entry.is_directory()) {
                std::string name = entry.path().filename().string();
                if (name.find(query) != std::string::npos) {
                    // Try to read its manifest
                    fs::path pkg_manifest = entry.path() / "eshkol.toml";
                    if (fs::exists(pkg_manifest)) {
                        Manifest pkg_m = load_manifest(entry.path());
                        std::cout << "  " << pkg_m.name << " v" << pkg_m.version;
                        if (!pkg_m.description.empty()) {
                            std::cout << " - " << pkg_m.description;
                        }
                        std::cout << std::endl;
                    } else {
                        std::cout << "  " << name << std::endl;
                    }
                    found = true;
                }
            }
        }
    }

    if (!found) {
        std::cout << "No packages found matching '" << query << "'" << std::endl;
        std::cout << "Registry: " << get_registry_url() << std::endl;
    }

    return 0;
}

int cmd_clean(int argc, char* argv[]) {
    fs::path dir = fs::current_path();
    fs::path build_dir = dir / "build";

    if (fs::exists(build_dir)) {
        fs::remove_all(build_dir);
        std::cout << "Removed build/" << std::endl;
    } else {
        std::cout << "Nothing to clean." << std::endl;
    }
    return 0;
}

int cmd_publish(int argc, char* argv[]) {
    fs::path dir = fs::current_path();
    Manifest m = load_manifest(dir);

    std::cout << "Publishing " << m.name << " v" << m.version << "..." << std::endl;
    std::cout << std::endl;
    std::cout << "To publish to the Eshkol registry:" << std::endl;
    std::cout << "  1. Push your code to a git repository" << std::endl;
    std::cout << "  2. Tag the release: git tag v" << m.version << std::endl;
    std::cout << "  3. Submit a PR to " << get_registry_url() << std::endl;
    std::cout << "     adding your package to the registry index" << std::endl;

    return 0;
}

// ============================================================================
// Main
// ============================================================================

void print_usage() {
    std::cout << "eshkol-pkg 1.1.0 - Eshkol Package Manager\n" << std::endl;
    std::cout << "Usage: eshkol-pkg <command> [args]\n" << std::endl;
    std::cout << "Commands:" << std::endl;
    std::cout << "  init              Create a new eshkol.toml project" << std::endl;
    std::cout << "  build             Compile the current project" << std::endl;
    std::cout << "  run               Build and run the project" << std::endl;
    std::cout << "  install           Install dependencies from eshkol.toml" << std::endl;
    std::cout << "  add <pkg> [ver]   Add a dependency" << std::endl;
    std::cout << "  remove <pkg>      Remove a dependency" << std::endl;
    std::cout << "  search <query>    Search the package registry" << std::endl;
    std::cout << "  publish           Publish a package" << std::endl;
    std::cout << "  clean             Remove build artifacts" << std::endl;
    std::cout << "\nEnvironment:" << std::endl;
    std::cout << "  ESHKOL_COMPILER   Path to eshkol-run (default: eshkol-run)" << std::endl;
    std::cout << "  ESHKOL_REGISTRY   Package registry URL" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string command = argv[1];

    if (command == "--version" || command == "-v") {
        std::cout << "eshkol-pkg 1.1.0" << std::endl;
        return 0;
    }
    if (command == "--help" || command == "-h") {
        print_usage();
        return 0;
    }

    // Dispatch subcommand
    int sub_argc = argc - 2;
    char** sub_argv = argv + 2;

    if (command == "init")    return cmd_init(sub_argc, sub_argv);
    if (command == "build")   return cmd_build(sub_argc, sub_argv);
    if (command == "run")     return cmd_run(sub_argc, sub_argv);
    if (command == "install") return cmd_install(sub_argc, sub_argv);
    if (command == "add")     return cmd_add(sub_argc, sub_argv);
    if (command == "remove")  return cmd_remove(sub_argc, sub_argv);
    if (command == "search")  return cmd_search(sub_argc, sub_argv);
    if (command == "publish") return cmd_publish(sub_argc, sub_argv);
    if (command == "clean")   return cmd_clean(sub_argc, sub_argv);

    std::cerr << "Unknown command: " << command << std::endl;
    print_usage();
    return 1;
}
