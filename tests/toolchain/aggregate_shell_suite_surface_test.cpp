#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

namespace {

std::string read_file(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

bool expect_contains(const std::string& haystack, const std::string& needle,
                     const std::string& label) {
    if (haystack.find(needle) == std::string::npos) {
        std::cerr << "missing: " << label << "\nneedle: " << needle << std::endl;
        return false;
    }
    return true;
}

bool expect_not_contains(const std::string& haystack, const std::string& needle,
                         const std::string& label) {
    if (haystack.find(needle) != std::string::npos) {
        std::cerr << "unexpected: " << label << "\nneedle: " << needle
                  << std::endl;
        return false;
    }
    return true;
}

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << std::endl;
    return 1;
}

bool check_aggregate_runner(const std::string& script, const std::string& label) {
    bool ok = true;

    ok = ok &&
         expect_contains(script, "require_regular_executable()",
                         label + " declares a regular executable guard") &&
         expect_contains(script, "! test -f \"$path\" || ! test -s \"$path\" || ! test -x \"$path\"",
                         label + " rejects non-regular executable paths") &&
         expect_contains(script, "BUILD_DIR=\"${BUILD_DIR:-build}\"",
                         label + " honors a configurable build directory") &&
         expect_contains(script, "require_regular_executable \"eshkol-run\" \"$BUILD_DIR/eshkol-run\"",
                         label + " validates the compiler executable before running suites") &&
         expect_contains(script, "validate_suite_script()",
                         label + " declares a suite script validator") &&
         expect_contains(script, "if ! test -e \"$path\"; then",
                         label + " treats absent suite scripts separately from unsafe scripts") &&
         expect_contains(script, "! test -f \"$path\" || ! test -s \"$path\"",
                         label + " rejects empty, symlinked, or non-regular suite scripts") &&
         expect_contains(script, "run_suite_script()",
                         label + " declares a suite launcher wrapper") &&
         expect_contains(script, "validate_suite_script \"$path\"",
                         label + " revalidates suite scripts inside the launcher") &&
         expect_contains(script, "run_suite_script \"$script_path\"",
                         label + " launches suites through the validated wrapper");

    ok = ok &&
         expect_not_contains(script, "if [ ! -f \"build/eshkol-run\" ]",
                             label + " must not use a plain compiler file probe") &&
         expect_not_contains(script, "if [ ! -f \"$script_path\" ]",
                             label + " must not use a plain suite script file probe") &&
         expect_not_contains(script, "bash \"$script_path\"",
                             label + " must not launch suite scripts without the wrapper");

    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        return fail("usage: aggregate_shell_suite_surface_test <source-root>");
    }

    const fs::path source_root = argv[1];
    const fs::path aggregate_runner = source_root / "scripts" / "run_all_tests.sh";
    const fs::path verbose_runner =
        source_root / "scripts" / "run_all_tests_with_output.sh";

    if (!fs::exists(aggregate_runner)) {
        return fail("run_all_tests.sh not found under source root");
    }
    if (!fs::exists(verbose_runner)) {
        return fail("run_all_tests_with_output.sh not found under source root");
    }

    bool ok = true;
    ok = check_aggregate_runner(read_file(aggregate_runner), "aggregate shell runner") && ok;
    ok = check_aggregate_runner(read_file(verbose_runner), "verbose aggregate shell runner") &&
         ok;

    if (!ok) {
        return 1;
    }

    std::cout << "PASS" << std::endl;
    return 0;
}
