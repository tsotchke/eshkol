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

int require_disjoint(
    const std::vector<std::pair<std::string, std::vector<std::string>>>& sets) {
    std::unordered_map<std::string, std::string> owner_by_source;
    for (const auto& [set_name, entries] : sets) {
        std::unordered_set<std::string> local_seen;
        for (const std::string& entry : entries) {
            if (!local_seen.insert(entry).second) {
                return fail(set_name + " contains duplicate source: " + entry);
            }

            const auto [it, inserted] = owner_by_source.emplace(entry, set_name);
            if (!inserted) {
                return fail("VM component source appears in both " + it->second +
                            " and " + set_name + ": " + entry);
            }
        }
    }
    return 0;
}

std::vector<std::string> parse_vm_unity_includes(const std::string& unity_contents) {
    std::vector<std::string> includes;
    const std::regex include_pattern(R"eshkol((^|\n)\s*#include\s+"((?:vm|eskb)_[^"]+\.c)")eshkol");
    auto begin = std::sregex_iterator(unity_contents.begin(), unity_contents.end(), include_pattern);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        includes.push_back("lib/backend/" + (*it)[2].str());
    }
    return includes;
}

int require_unity_classification(
    const std::vector<std::string>& unity_includes,
    const std::vector<std::pair<std::string, std::vector<std::string>>>& sets) {
    std::unordered_map<std::string, std::string> owner_by_source;
    for (const auto& [set_name, entries] : sets) {
        for (const std::string& entry : entries) {
            owner_by_source.emplace(entry, set_name);
        }
    }

    std::unordered_set<std::string> included;
    for (const std::string& entry : unity_includes) {
        included.insert(entry);
        if (owner_by_source.find(entry) == owner_by_source.end()) {
            return fail("VM unity include is not assigned to a component family: " + entry);
        }
    }

    for (const auto& [entry, set_name] : owner_by_source) {
        if (included.find(entry) == included.end()) {
            return fail(set_name + " source is not included by eshkol_vm.c: " + entry);
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        return fail("usage: vm_source_boundary_test <source-root>");
    }

    const std::filesystem::path source_root = argv[1];
    const std::string cmake_contents = read_file(source_root / "CMakeLists.txt");
    if (cmake_contents.empty()) {
        return fail("failed to read CMakeLists.txt");
    }

    const std::vector<std::string> vm_core_src =
        parse_cmake_list(cmake_contents, "ESHKOL_VM_CORE_COMPONENT_SRC");
    const std::vector<std::string> vm_hosted_src =
        parse_cmake_list(cmake_contents, "ESHKOL_VM_HOSTED_COMPONENT_SRC");
    const std::vector<std::string> vm_toolchain_src =
        parse_cmake_list(cmake_contents, "ESHKOL_VM_TOOLCHAIN_COMPONENT_SRC");
    const std::vector<std::string> vm_test_src =
        parse_cmake_list(cmake_contents, "ESHKOL_VM_TEST_COMPONENT_SRC");
    const std::vector<std::string> vm_unity_src =
        parse_cmake_list(cmake_contents, "ESHKOL_VM_FULL_UNITY_SRC");

    if (vm_core_src.empty()) return fail("VM core component source set is empty or missing");
    if (vm_hosted_src.empty()) return fail("VM hosted component source set is empty or missing");
    if (vm_toolchain_src.empty()) return fail("VM toolchain component source set is empty or missing");
    if (vm_test_src.empty()) return fail("VM test component source set is empty or missing");
    if (vm_unity_src.empty()) return fail("VM unity source set is empty or missing");

    if (const int rc = require_entries(vm_core_src,
            {"lib/backend/vm_core.c", "lib/backend/vm_run.c",
             "lib/backend/vm_complex.c", "lib/backend/vm_rational.c",
             "lib/backend/vm_bignum.c", "lib/backend/vm_dual.c",
             "lib/backend/vm_hyperdual.c", "lib/backend/vm_autodiff.c",
             "lib/backend/vm_tensor.c", "lib/backend/vm_tensor_ops.c",
             "lib/backend/vm_logic.c", "lib/backend/vm_inference.c",
             "lib/backend/vm_workspace.c", "lib/backend/vm_string.c",
             "lib/backend/vm_hashtable.c", "lib/backend/vm_bytevector.c",
             "lib/backend/vm_multivalue.c", "lib/backend/vm_error.c",
             "lib/backend/vm_parameter.c", "lib/backend/vm_geometric.c",
             "lib/backend/vm_symbolic_ad.c"},
            "VM core component source set")) {
        return rc;
    }
    if (const int rc = require_entries(vm_hosted_src,
            {"lib/backend/eskb_writer.c", "lib/backend/eskb_reader.c",
             "lib/backend/vm_io.c", "lib/backend/vm_model_io.c",
             "lib/backend/vm_parallel.c", "lib/backend/vm_native.c"},
            "VM hosted component source set")) {
        return rc;
    }
    if (const int rc = require_entries(vm_toolchain_src,
            {"lib/backend/vm_parser.c", "lib/backend/vm_macro.c",
             "lib/backend/vm_compiler.c", "lib/backend/vm_peephole.c"},
            "VM toolchain component source set")) {
        return rc;
    }
    if (const int rc = require_entries(vm_test_src,
            {"lib/backend/vm_tests.c"},
            "VM test component source set")) {
        return rc;
    }
    if (const int rc = require_entries(vm_unity_src,
            {"lib/backend/eshkol_vm.c"},
            "VM unity source set")) {
        return rc;
    }

    if (const int rc = require_disjoint({
            {"VM core", vm_core_src},
            {"VM hosted", vm_hosted_src},
            {"VM toolchain", vm_toolchain_src},
            {"VM test", vm_test_src},
        })) {
        return rc;
    }

    if (const int rc = require_sources_exist(source_root, vm_core_src, "VM core")) return rc;
    if (const int rc = require_sources_exist(source_root, vm_hosted_src, "VM hosted")) return rc;
    if (const int rc = require_sources_exist(source_root, vm_toolchain_src, "VM toolchain")) return rc;
    if (const int rc = require_sources_exist(source_root, vm_test_src, "VM test")) return rc;
    if (const int rc = require_sources_exist(source_root, vm_unity_src, "VM unity")) return rc;

    if (!contains_marker(cmake_contents,
                         "add_library(eshkol-vm-unity-obj OBJECT ${ESHKOL_VM_UNITY_SRC})")) {
        return fail("VM unity source is not compiled through eshkol-vm-unity-obj");
    }
    if (!contains_marker(cmake_contents,
                         "set(ESHKOL_VM_UNITY_OBJECTS $<TARGET_OBJECTS:eshkol-vm-unity-obj>)")) {
        return fail("eshkol-vm-unity-obj objects are not captured for eshkol-static");
    }
    if (!contains_marker(cmake_contents, "${ESHKOL_VM_UNITY_OBJECTS}")) {
        return fail("eshkol-static does not include VM unity object output");
    }
    if (contains_marker(cmake_contents, "list(APPEND LIB_SRC lib/backend/eshkol_vm.c)")) {
        return fail("VM unity source should not be appended directly to LIB_SRC");
    }

    const std::string unity_contents = read_file(source_root / "lib/backend/eshkol_vm.c");
    if (unity_contents.empty()) {
        return fail("failed to read lib/backend/eshkol_vm.c");
    }
    const std::vector<std::string> unity_includes = parse_vm_unity_includes(unity_contents);
    if (unity_includes.empty()) {
        return fail("eshkol_vm.c has no parsed VM component includes");
    }
    if (const int rc = require_unity_classification(unity_includes, {
            {"VM core", vm_core_src},
            {"VM hosted", vm_hosted_src},
            {"VM toolchain", vm_toolchain_src},
            {"VM test", vm_test_src},
        })) {
        return rc;
    }

    const std::vector<std::string_view> hosted_markers = {
        "FILE*",
        "fopen(",
        "fread(",
        "fwrite(",
        "fclose(",
        "getenv(",
        "std::getenv(",
        "pthread_",
        "fork(",
        "execvp(",
        "execve(",
        "execlp(",
        "socket(",
        "bind(",
        "listen(",
        "accept(",
        "poll(",
        "select(",
        "opendir(",
        "readdir(",
        "mkdir(",
        "chdir(",
        "getcwd(",
        "dlopen(",
        "mmap(",
        "tmpfile(",
        "CreateProcess",
        "WaitForSingleObject",
    };

    const std::vector<std::pair<std::string, std::regex>> hosted_include_regexes = {
        {"#include <dirent.h>", std::regex(R"((^|\n)\s*#include\s*<dirent\.h>)")},
        {"#include <dlfcn.h>", std::regex(R"((^|\n)\s*#include\s*<dlfcn\.h>)")},
        {"#include <fcntl.h>", std::regex(R"((^|\n)\s*#include\s*<fcntl\.h>)")},
        {"#include <glob.h>", std::regex(R"((^|\n)\s*#include\s*<glob\.h>)")},
        {"#include <netdb.h>", std::regex(R"((^|\n)\s*#include\s*<netdb\.h>)")},
        {"#include <poll.h>", std::regex(R"((^|\n)\s*#include\s*<poll\.h>)")},
        {"#include <pthread.h>", std::regex(R"((^|\n)\s*#include\s*<pthread\.h>)")},
        {"#include <signal.h>", std::regex(R"((^|\n)\s*#include\s*<signal\.h>)")},
        {"#include <sys/mman.h>", std::regex(R"((^|\n)\s*#include\s*<sys/mman\.h>)")},
        {"#include <sys/socket.h>", std::regex(R"((^|\n)\s*#include\s*<sys/socket\.h>)")},
        {"#include <sys/wait.h>", std::regex(R"((^|\n)\s*#include\s*<sys/wait\.h>)")},
        {"#include <unistd.h>", std::regex(R"((^|\n)\s*#include\s*<unistd\.h>)")},
        {"#include <windows.h>", std::regex(R"((^|\n)\s*#include\s*<windows\.h>)")},
    };

    for (const std::string& entry : vm_core_src) {
        const std::string contents = read_file(source_root / entry);
        if (contents.empty()) {
            return fail("failed to read VM core source: " + entry);
        }
        for (const std::string_view marker : hosted_markers) {
            if (contains_marker(contents, marker)) {
                return fail("VM core source " + entry +
                            " contains hosted-only marker: " + std::string(marker));
            }
        }
        for (const auto& [label, pattern] : hosted_include_regexes) {
            if (contains_regex(contents, pattern)) {
                return fail("VM core source " + entry +
                            " contains hosted-only include: " + label);
            }
        }
    }

    std::cout << "PASS" << std::endl;
    return 0;
}
