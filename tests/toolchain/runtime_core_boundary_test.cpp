#include <filesystem>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <regex>
#include <string>
#include <string_view>
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
    std::ifstream input(path);
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

bool contains_marker(const std::string& contents, std::string_view marker) {
    return contents.find(marker) != std::string::npos;
}

bool contains_regex(const std::string& contents, const std::regex& pattern) {
    return std::regex_search(contents, pattern);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        return fail("expected source directory argument");
    }

    const std::filesystem::path source_root = argv[1];
    const std::filesystem::path cmake_path = source_root / "CMakeLists.txt";
    const std::string cmake_contents = read_file(cmake_path);
    if (cmake_contents.empty()) {
        return fail("failed to read CMakeLists.txt");
    }

    const std::vector<std::string> runtime_core_src =
        parse_cmake_list(cmake_contents, "ESHKOL_RUNTIME_CORE_SRC");
    if (runtime_core_src.empty()) {
        return fail("runtime core source set is empty or missing");
    }

    const std::vector<std::string> runtime_hosted_src =
        parse_cmake_list(cmake_contents, "ESHKOL_RUNTIME_HOSTED_SRC");
    if (runtime_hosted_src.empty()) {
        return fail("runtime hosted source set is empty or missing");
    }

    const std::vector<std::string> runtime_freestanding_src =
        parse_cmake_list(cmake_contents, "ESHKOL_RUNTIME_FREESTANDING_SRC");
    if (runtime_freestanding_src.empty()) {
        return fail("runtime freestanding source set is empty or missing");
    }

    bool found_arena_memory = false;
    bool found_resource_limits_core = false;
    for (const std::string& entry : runtime_core_src) {
        if (entry == "lib/core/arena_memory.cpp") {
            found_arena_memory = true;
        }
        if (entry == "lib/core/resource_limits_core.cpp") {
            found_resource_limits_core = true;
        }
    }
    if (!found_arena_memory) {
        return fail("runtime core source set does not include lib/core/arena_memory.cpp");
    }
    if (!found_resource_limits_core) {
        return fail("runtime core source set does not include lib/core/resource_limits_core.cpp");
    }

    bool found_resource_limits_hosted = false;
    for (const std::string& entry : runtime_hosted_src) {
        if (entry == "lib/core/resource_limits_hosted.cpp") {
            found_resource_limits_hosted = true;
            break;
        }
    }
    if (!found_resource_limits_hosted) {
        return fail("runtime hosted source set does not include lib/core/resource_limits_hosted.cpp");
    }

    const std::vector<std::string> runtime_compat_src =
        parse_cmake_list(cmake_contents, "ESHKOL_RUNTIME_COMPAT_SRC");
    bool found_resource_limits_compat = false;
    for (const std::string& entry : runtime_compat_src) {
        if (entry == "lib/core/resource_limits.cpp") {
            found_resource_limits_compat = true;
            break;
        }
    }
    if (!found_resource_limits_compat) {
        return fail("runtime compatibility source set does not retain lib/core/resource_limits.cpp");
    }
    for (const std::string& entry : runtime_core_src) {
        if (entry == "lib/core/resource_limits.cpp") {
            return fail("legacy resource_limits.cpp should not be compiled in runtime core set");
        }
    }
    for (const std::string& entry : runtime_hosted_src) {
        if (entry == "lib/core/resource_limits.cpp") {
            return fail("legacy resource_limits.cpp should not be compiled in runtime hosted set");
        }
    }
    for (const std::string& entry : runtime_freestanding_src) {
        if (entry == "lib/core/resource_limits.cpp") {
            return fail("legacy resource_limits.cpp should not be compiled in runtime freestanding set");
        }
    }

    const std::vector<std::string> split_pending_src =
        parse_cmake_list(cmake_contents, "ESHKOL_RUNTIME_SPLIT_PENDING_SRC");
    if (!split_pending_src.empty()) {
        return fail("runtime split-pending source set should be empty");
    }

    std::unordered_set<std::string> runtime_core_entries(runtime_core_src.begin(),
                                                         runtime_core_src.end());
    for (const std::string& entry : runtime_hosted_src) {
        if (runtime_core_entries.find(entry) != runtime_core_entries.end()) {
            return fail("runtime source appears in both core and hosted sets: " + entry);
        }
    }
    std::unordered_set<std::string> runtime_hosted_entries(runtime_hosted_src.begin(),
                                                           runtime_hosted_src.end());
    for (const std::string& entry : runtime_freestanding_src) {
        if (runtime_core_entries.find(entry) != runtime_core_entries.end()) {
            return fail("runtime source appears in both core and freestanding sets: " + entry);
        }
        if (runtime_hosted_entries.find(entry) != runtime_hosted_entries.end()) {
            return fail("runtime source appears in both hosted and freestanding sets: " + entry);
        }
    }

    const std::vector<std::string> forbidden_markers = {
        "logger.h",
        "platform_runtime.h",
        "runtime_exports.h",
        "std::filesystem",
        "std::thread",
        "eshkol_error(",
        "eshkol_warn(",
        "eshkol_debug(",
        "eshkol_printf(",
        "CreateProcessW(",
        "execvp(",
        "fork(",
        "fprintf(",
        "fgetc(",
        "getenv(",
        "std::getenv(",
        "open(",
        "tmpfile(",
        "fmemopen(",
        "open_memstream(",
        "abort(",
        "std::abort(",
        "waitpid(",
        "pthread_",
        "sigaction("
    };

    const std::vector<std::pair<std::string, std::regex>> forbidden_regexes = {
        {"#include <filesystem>", std::regex(R"((^|\n)\s*#include\s*<filesystem>)")},
        {"#include <fstream>", std::regex(R"((^|\n)\s*#include\s*<fstream>)")},
        {"#include <iostream>", std::regex(R"((^|\n)\s*#include\s*<iostream>)")},
        {"#include <pthread.h>", std::regex(R"((^|\n)\s*#include\s*<pthread\.h>)")},
        {"#include <signal.h>", std::regex(R"((^|\n)\s*#include\s*<signal\.h>)")},
        {"#include <sys/wait.h>", std::regex(R"((^|\n)\s*#include\s*<sys/wait\.h>)")},
        {"#include <thread>", std::regex(R"((^|\n)\s*#include\s*<thread>)")},
        {"#include <unistd.h>", std::regex(R"((^|\n)\s*#include\s*<unistd\.h>)")},
        {"stdin", std::regex(R"(\bstdin\b)")},
        {"stdout", std::regex(R"(\bstdout\b)")},
        {"stderr", std::regex(R"(\bstderr\b)")}
    };

    const std::vector<std::string> freestanding_forbidden_markers = {
        "logger.h",
        "platform_runtime.h",
        "runtime_exports.h",
        "std::filesystem",
        "std::thread",
        "eshkol_error(",
        "eshkol_warn(",
        "eshkol_debug(",
        "eshkol_printf(",
        "CreateProcessW(",
        "execvp(",
        "fork(",
        "fprintf(",
        "fgetc(",
        "getenv(",
        "std::getenv(",
        "open(",
        "tmpfile(",
        "fmemopen(",
        "open_memstream(",
        "abort(",
        "std::abort(",
        "waitpid(",
        "pthread_",
        "sigaction("
    };

    for (const std::string& entry : runtime_core_src) {
        const std::filesystem::path source_path = source_root / entry;
        if (!std::filesystem::exists(source_path)) {
            return fail("runtime core source does not exist: " + entry);
        }
        const std::string contents = read_file(source_path);
        if (contents.empty()) {
            return fail("failed to read runtime core source: " + entry);
        }

        if (source_path.filename().string().find("_hosted") != std::string::npos) {
            return fail("runtime core source is still classified as hosted: " + entry);
        }

        for (std::string_view marker : forbidden_markers) {
            if (contains_marker(contents, marker)) {
                return fail("runtime core source " + entry +
                            " contains hosted marker: " + std::string(marker));
            }
        }

        for (const auto& [label, pattern] : forbidden_regexes) {
            if (contains_regex(contents, pattern)) {
                return fail("runtime core source " + entry +
                            " contains hosted marker: " + label);
            }
        }
    }

    for (const std::string& entry : runtime_hosted_src) {
        const std::filesystem::path source_path = source_root / entry;
        if (!std::filesystem::exists(source_path)) {
            return fail("runtime hosted source does not exist: " + entry);
        }
    }

    for (const std::string& entry : runtime_freestanding_src) {
        const std::filesystem::path source_path = source_root / entry;
        if (!std::filesystem::exists(source_path)) {
            return fail("runtime freestanding source does not exist: " + entry);
        }
        const std::string contents = read_file(source_path);
        if (contents.empty()) {
            return fail("failed to read runtime freestanding source: " + entry);
        }

        if (source_path.filename().string().find("_hosted") != std::string::npos) {
            return fail("runtime freestanding source is still classified as hosted: " + entry);
        }

        for (std::string_view marker : freestanding_forbidden_markers) {
            if (contains_marker(contents, marker)) {
                return fail("runtime freestanding source " + entry +
                            " contains hosted marker: " + std::string(marker));
            }
        }

        for (const auto& [label, pattern] : forbidden_regexes) {
            if (contains_regex(contents, pattern)) {
                return fail("runtime freestanding source " + entry +
                            " contains hosted marker: " + label);
            }
        }
    }

    std::cout << "PASS" << std::endl;
    return 0;
}
