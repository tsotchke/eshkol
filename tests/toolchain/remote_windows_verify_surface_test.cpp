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

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        return fail("usage: remote_windows_verify_surface_test <source-root>");
    }

    const fs::path source_root = argv[1];
    const fs::path script_path =
        source_root / "scripts" / "remote_windows_verify.sh";
    if (!fs::exists(script_path)) {
        return fail("remote_windows_verify.sh not found under source root");
    }

    const std::string script = read_file(script_path);
    bool ok = true;

    ok = ok &&
         expect_contains(script, "GENERATOR=\"${GENERATOR:-Visual Studio 17 2022}\"",
                         "remote Windows verifier defaults to Visual Studio 2022") &&
         expect_contains(script, "TOOLSET=\"${TOOLSET:-ClangCL}\"",
                         "remote Windows verifier defaults to ClangCL") &&
         expect_contains(script, "ARCH=\"${ARCH:-x64}\"",
                         "remote Windows verifier defaults to x64") &&
         expect_contains(script, "LLVM_VERSION=\"${LLVM_VERSION:-21}\"",
                         "remote Windows verifier pins LLVM 21 by default") &&
         expect_contains(script, "return (Join-Path $env:USERPROFILE \"projects\\eshkol\")",
                         "remote Windows verifier has a user-profile checkout default") &&
         expect_contains(script, "Test-Path -LiteralPath $RepoDir -PathType Container",
                         "remote Windows verifier checks the checkout directory literally") &&
         expect_contains(script, "git @(\"fetch\", \"origin\", \"master\")",
                         "remote Windows verifier fetches origin/master") &&
         expect_contains(script, "git @(\"merge\", \"--ff-only\", $FetchRefArg)",
                         "remote Windows verifier fast-forwards without merge commits") &&
         expect_contains(script, "\"-DESHKOL_BUILD_TESTS=ON\"",
                         "remote Windows verifier enables CTest targets") &&
         expect_contains(script, "\"-DESHKOL_REQUIRED_LLVM_MAJOR=$LLVMVersionArg\"",
                         "remote Windows verifier passes the required LLVM version") &&
         expect_contains(script, "\"-DLLVM_DIR=$LLVMDirArg\"",
                         "remote Windows verifier supports explicit LLVM_DIR") &&
         expect_contains(script,
                         "\"--target\", \"eshkol-run\", \"eshkol-repl\", \"stdlib\", \"windows_suite_surface_test\"",
                         "remote Windows verifier builds the native compiler, REPL, stdlib, and surface guard") &&
         expect_contains(script, "\"windows_suite_surface_test\"",
                         "remote Windows verifier runs the Windows CTest surface guard") &&
         expect_contains(script, "Join-Path $RepoDir \"scripts\\run_all_tests.ps1\"",
                         "remote Windows verifier invokes the existing PowerShell suite") &&
         expect_contains(script, "\"-Mode\", \"windows-lite\"",
                         "remote Windows verifier runs the bounded Windows smoke suite") &&
         expect_contains(script, "git @(\"diff\", \"--check\")",
                         "remote Windows verifier checks whitespace after validation") &&
         expect_contains(script,
                         "powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Command -",
                         "remote Windows verifier sends a PowerShell script over SSH");

    ok = ok &&
         expect_not_contains(script, "git reset --hard",
                             "remote Windows verifier must not reset remote worktrees") &&
         expect_not_contains(script, "Remove-Item -Recurse",
                             "remote Windows verifier must not delete remote trees") &&
         expect_not_contains(script, "-Mode\", \"all\"",
                             "remote Windows verifier should stay bounded to windows-lite by default");

    if (!ok) {
        return 1;
    }

    std::cout << "PASS" << std::endl;
    return 0;
}
