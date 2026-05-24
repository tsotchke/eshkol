#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << std::endl;
    return 1;
}

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return {};
    }
    return std::string((std::istreambuf_iterator<char>(input)),
                       std::istreambuf_iterator<char>());
}

std::vector<std::string> parse_cmake_list(const std::string& cmake_contents,
                                          const std::string& list_name) {
    const std::string marker = "set(" + list_name;
    const std::size_t start = cmake_contents.find(marker);
    if (start == std::string::npos) {
        return {};
    }

    const std::size_t line_end = cmake_contents.find('\n', start);
    const std::string first_line = cmake_contents.substr(
        start, line_end == std::string::npos ? std::string::npos : line_end - start);
    if (first_line.find(')') != std::string::npos) {
        return {};
    }

    std::vector<std::string> entries;
    std::size_t cursor = line_end == std::string::npos ? cmake_contents.size() : line_end + 1;
    while (cursor < cmake_contents.size()) {
        const std::size_t next_end = cmake_contents.find('\n', cursor);
        const std::string raw_line = cmake_contents.substr(
            cursor, next_end == std::string::npos ? std::string::npos : next_end - cursor);
        const std::string line = trim(raw_line);
        if (line == ")") {
            break;
        }
        if (!line.empty() && line[0] != '#') {
            entries.push_back(line);
        }
        if (next_end == std::string::npos) {
            break;
        }
        cursor = next_end + 1;
    }
    return entries;
}

bool has_entry(const std::vector<std::string>& entries, const std::string& expected) {
    for (const std::string& entry : entries) {
        if (entry == expected) {
            return true;
        }
    }
    return false;
}

bool contains_marker(const std::string& contents, std::string_view marker) {
    return contents.find(marker) != std::string::npos;
}

bool contains_regex(const std::string& contents, const std::regex& pattern) {
    return std::regex_search(contents, pattern);
}

int require_entries(const std::vector<std::string>& entries,
                    const std::vector<std::string>& expected,
                    const std::string& set_name) {
    for (const std::string& entry : expected) {
        if (!has_entry(entries, entry)) {
            return fail(set_name + " is missing expected source: " + entry);
        }
    }
    return 0;
}

int require_sources_exist(const std::filesystem::path& source_root,
                          const std::vector<std::string>& entries,
                          const std::string& set_name) {
    for (const std::string& entry : entries) {
        if (!std::filesystem::exists(source_root / entry)) {
            return fail(set_name + " source does not exist: " + entry);
        }
    }
    return 0;
}

int require_disjoint(const std::vector<std::pair<std::string, std::vector<std::string>>>& sets) {
    std::unordered_map<std::string, std::string> owner_by_source;
    for (const auto& [set_name, entries] : sets) {
        std::unordered_set<std::string> local_seen;
        for (const std::string& entry : entries) {
            if (!local_seen.insert(entry).second) {
                return fail(set_name + " contains duplicate source: " + entry);
            }

            const auto [it, inserted] = owner_by_source.emplace(entry, set_name);
            if (!inserted) {
                return fail("runtime source appears in both " + it->second +
                            " and " + set_name + ": " + entry);
            }
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        return fail("usage: runtime_core_boundary_test <source-root>");
    }

    const std::filesystem::path source_root = argv[1];
    const std::string cmake_contents = read_file(source_root / "CMakeLists.txt");
    if (cmake_contents.empty()) {
        return fail("failed to read CMakeLists.txt");
    }

    const std::vector<std::string> runtime_core_src =
        parse_cmake_list(cmake_contents, "ESHKOL_RUNTIME_CORE_SRC");
    const std::vector<std::string> runtime_hosted_src =
        parse_cmake_list(cmake_contents, "ESHKOL_RUNTIME_HOSTED_SRC");
    const std::vector<std::string> split_pending_src =
        parse_cmake_list(cmake_contents, "ESHKOL_RUNTIME_SPLIT_PENDING_SRC");

    if (runtime_core_src.empty()) {
        return fail("runtime core source set is empty or missing");
    }
    if (runtime_hosted_src.empty()) {
        return fail("runtime hosted source set is empty or missing");
    }
    if (split_pending_src.empty()) {
        return fail("runtime split-pending source set is empty or missing");
    }

    if (const int rc = require_entries(runtime_core_src,
            {"lib/core/ad_tape_builtins.c", "lib/core/bignum.cpp",
             "lib/core/rational.cpp", "lib/core/runtime.cpp",
             "lib/core/runtime_autodiff.cpp",
             "lib/core/runtime_bytevector.cpp",
             "lib/core/runtime_closure_reflection.cpp",
             "lib/core/runtime_continuations.cpp",
             "lib/core/runtime_hash_table.cpp",
             "lib/core/runtime_list_helpers.cpp",
             "lib/core/runtime_shared_memory.cpp",
             "lib/core/runtime_string.cpp",
             "lib/core/runtime_tensor_alloc.cpp",
             "lib/core/runtime_tensor_index.cpp",
             "lib/core/runtime_tensor_fill.cpp",
             "lib/core/runtime_tensor_math.cpp"},
            "runtime core source set")) {
        return rc;
    }
    if (const int rc = require_entries(runtime_hosted_src,
            {"lib/backend/thread_pool.cpp", "lib/core/config.cpp",
             "lib/core/logger.cpp", "lib/core/platform_runtime.cpp",
             "lib/core/resource_limits.cpp",
             "lib/core/runtime_arena_sync_hosted.cpp",
             "lib/core/runtime_display_hosted.cpp",
             "lib/core/runtime_errors_hosted.cpp",
             "lib/core/runtime_exceptions_hosted.cpp",
             "lib/core/runtime_exports_hosted.cpp",
             "lib/core/runtime_lifecycle_hosted.cpp",
             "lib/core/runtime_operations_hosted.cpp",
             "lib/core/runtime_parameters_hosted.cpp",
             "lib/core/runtime_reader_hosted.cpp",
             "lib/core/runtime_shutdown_hooks_hosted.cpp",
             "lib/core/runtime_signals_hosted.cpp",
             "lib/core/runtime_stack_hosted.cpp",
             "lib/core/runtime_string_ports_hosted.cpp",
             "lib/core/system_builtins.c"},
            "runtime hosted source set")) {
        return rc;
    }
    if (const int rc = require_entries(split_pending_src,
            {"lib/core/arena_memory.cpp"},
            "runtime split-pending source set")) {
        return rc;
    }

    if (const int rc = require_disjoint({
            {"runtime core", runtime_core_src},
            {"runtime hosted", runtime_hosted_src},
            {"runtime split-pending", split_pending_src},
        })) {
        return rc;
    }

    if (!contains_marker(cmake_contents,
                         "add_library(eshkol-runtime-core-obj OBJECT ${ESHKOL_RUNTIME_CORE_SRC})")) {
        return fail("runtime core source set is not compiled through eshkol-runtime-core-obj");
    }
    if (!contains_marker(cmake_contents,
                         "add_library(eshkol-runtime-hosted-obj OBJECT ${ESHKOL_RUNTIME_HOSTED_SRC})")) {
        return fail("runtime hosted source set is not compiled through eshkol-runtime-hosted-obj");
    }
    if (!contains_marker(cmake_contents,
                         "add_library(eshkol-runtime-split-pending-obj OBJECT ${ESHKOL_RUNTIME_SPLIT_PENDING_SRC})")) {
        return fail("runtime split-pending source set is not compiled through eshkol-runtime-split-pending-obj");
    }

    if (const int rc = require_sources_exist(source_root, runtime_core_src, "runtime core")) {
        return rc;
    }
    if (const int rc = require_sources_exist(source_root, runtime_hosted_src, "runtime hosted")) {
        return rc;
    }
    if (const int rc = require_sources_exist(source_root, split_pending_src, "runtime split-pending")) {
        return rc;
    }

    const std::vector<std::string_view> forbidden_markers = {
        "platform_runtime.h",
        "runtime_exports.h",
        "std::filesystem",
        "std::thread",
        "CreateProcess",
        "execvp(",
        "fork(",
        "waitpid(",
        "getenv(",
        "std::getenv(",
        "tmpfile(",
        "fmemopen(",
        "open_memstream(",
        "socket(",
        "bind(",
        "listen(",
        "accept(",
        "pthread_",
    };

    const std::vector<std::pair<std::string, std::regex>> forbidden_regexes = {
        {"#include <filesystem>", std::regex(R"((^|\n)\s*#include\s*<filesystem>)")},
        {"#include <fstream>", std::regex(R"((^|\n)\s*#include\s*<fstream>)")},
        {"#include <iostream>", std::regex(R"((^|\n)\s*#include\s*<iostream>)")},
        {"#include <pthread.h>", std::regex(R"((^|\n)\s*#include\s*<pthread\.h>)")},
        {"#include <signal.h>", std::regex(R"((^|\n)\s*#include\s*<signal\.h>)")},
        {"#include <sys/socket.h>", std::regex(R"((^|\n)\s*#include\s*<sys/socket\.h>)")},
        {"#include <sys/wait.h>", std::regex(R"((^|\n)\s*#include\s*<sys/wait\.h>)")},
        {"#include <thread>", std::regex(R"((^|\n)\s*#include\s*<thread>)")},
        {"#include <unistd.h>", std::regex(R"((^|\n)\s*#include\s*<unistd\.h>)")},
        {"#include <winsock2.h>", std::regex(R"((^|\n)\s*#include\s*<winsock2\.h>)")},
    };

    for (const std::string& entry : runtime_core_src) {
        const std::filesystem::path source_path = source_root / entry;
        const std::string contents = read_file(source_path);
        if (contents.empty()) {
            return fail("failed to read runtime core source: " + entry);
        }

        if (source_path.filename().string().find("_hosted") != std::string::npos) {
            return fail("runtime core source is named as hosted: " + entry);
        }

        for (const std::string_view marker : forbidden_markers) {
            if (contains_marker(contents, marker)) {
                return fail("runtime core source " + entry +
                            " contains hosted-only marker: " + std::string(marker));
            }
        }

        for (const auto& [label, pattern] : forbidden_regexes) {
            if (contains_regex(contents, pattern)) {
                return fail("runtime core source " + entry +
                            " contains hosted-only marker: " + label);
            }
        }
    }

    std::cout << "PASS" << std::endl;
    return 0;
}
