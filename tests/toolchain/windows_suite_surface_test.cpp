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
        return fail("usage: windows_suite_surface_test <source-root>");
    }

    const fs::path source_root = argv[1];
    const fs::path script_path = source_root / "scripts" / "run_all_tests.ps1";
    if (!fs::exists(script_path)) {
        return fail("run_all_tests.ps1 not found under source root");
    }

    const std::string script = read_file(script_path);
    const std::string arena_header =
        read_file(source_root / "lib" / "core" / "arena_memory.h");
    const std::string runtime_arena_core =
        read_file(source_root / "lib" / "core" / "runtime_arena_core.cpp");
    const std::string runtime_regions =
        read_file(source_root / "lib" / "core" / "runtime_regions.cpp");
    bool ok = true;

    ok = ok &&
         expect_contains(script, "function Get-RegularFileItem",
                         "Windows suite declares regular-file resolver") &&
         expect_contains(script, "Get-Item -LiteralPath $Path -Force -ErrorAction Stop",
                         "Windows suite resolves file candidates without wildcard expansion") &&
         expect_contains(script, "function Test-RegularFile",
                         "Windows suite declares regular-file predicate") &&
         expect_contains(script,
                         "($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0",
                         "Windows suite rejects reparse-point file artifacts") &&
         expect_contains(script, "function Remove-RegularFileIfPresent",
                         "Windows suite declares guarded output cleanup") &&
         expect_contains(script, "Remove-Item -LiteralPath $item.FullName -Force -ErrorAction Stop",
                         "Windows suite cleanup removes only the resolved regular file") &&
         expect_contains(script, "function Start-RegularFileProcess",
                         "Windows suite declares guarded process launcher") &&
         expect_contains(script, "Refusing to start missing or non-regular executable",
                         "Windows suite guarded launcher rejects non-regular executables") &&
         expect_contains(script, "Test-RegularFile -Path (Join-Path $binDir \"eshkol-run.exe\")",
                         "Windows suite build resolver requires a regular eshkol-run.exe") &&
         expect_contains(script, "Remove-RegularFileIfPresent -Path @($exePath, $OutputBase)",
                         "Windows suite compile cleanup rejects non-regular output artifacts") &&
         expect_contains(script,
                         "Success  = ($captured.ExitCode -eq 0 -and (Test-RegularFile -Path $exePath))",
                         "Windows suite compile success requires a regular executable") &&
         expect_contains(script, "Test-RegularFile -Path $compile.ExePath",
                         "Windows suite expected-error path checks regular executables") &&
         expect_contains(script, "Test-RegularFile -Path $llvmCommand.Source",
                         "Windows suite PATH llvm-config fallback rejects non-regular commands") &&
         expect_contains(script, "Test-RegularFile -Path $script:EshkolServer",
                         "Windows suite server runner requires a regular executable") &&
         expect_contains(script, "Start-RegularFileProcess -FilePath $script:EshkolServer",
                         "Windows suite starts the web server through the guarded launcher") &&
         expect_contains(script, "Test-RegularFile -Path $wasmPath",
                         "Windows suite WASM validation requires a regular output file") &&
         expect_contains(script, "Remove-RegularFileIfPresent -Path @($serverStdout, $serverStderr)",
                         "Windows suite server-log cleanup rejects non-regular artifacts") &&
         expect_contains(script, "Test-RegularFile -Path $xlaBin",
                         "Windows suite XLA binary probe requires a regular executable") &&
         expect_contains(script, "Test-RegularFile -Path $path",
                         "Windows suite build-artifact probes require regular files");

    ok = ok &&
         expect_not_contains(script, "Test-Path ",
                             "Windows suite should not use plain Test-Path artifact probes") &&
         expect_not_contains(script, "Remove-Item $",
                             "Windows suite should not remove variable paths without the guard helper") &&
         expect_not_contains(script, "$server = Start-Process",
                             "Windows suite should not start the web server without the guard helper");

    ok = ok &&
         expect_contains(arena_header, "#if defined(__GNUC__) && !defined(_WIN32)",
                         "runtime weak-linkage macro excludes Windows COFF") &&
         expect_contains(arena_header, "#define ESHKOL_RUNTIME_WEAK __attribute__((weak))",
                         "runtime weak-linkage macro preserves ELF/Mach-O weak defaults") &&
         expect_contains(runtime_arena_core,
                         "ESHKOL_RUNTIME_WEAK int32_t __eshkol_argc = 0;",
                         "runtime argc default uses portable weak-linkage macro") &&
         expect_contains(runtime_arena_core,
                         "ESHKOL_RUNTIME_WEAK char** __eshkol_argv = nullptr;",
                         "runtime argv default uses portable weak-linkage macro") &&
         expect_contains(runtime_regions,
                         "ESHKOL_RUNTIME_WEAK arena_t* __global_arena = nullptr;",
                         "runtime arena default uses portable weak-linkage macro") &&
         expect_not_contains(runtime_arena_core,
                             "__attribute__((weak)) int32_t __eshkol_argc",
                             "runtime argc default should not use direct weak attribute") &&
         expect_not_contains(runtime_arena_core,
                             "__attribute__((weak)) char** __eshkol_argv",
                             "runtime argv default should not use direct weak attribute") &&
         expect_not_contains(runtime_regions,
                             "__attribute__((weak)) arena_t* __global_arena",
                             "runtime arena default should not use direct weak attribute");

    if (!ok) {
        return 1;
    }

    std::cout << "PASS" << std::endl;
    return 0;
}
